"""Train DQN, PPO, PPO-Lagrangian or SAC on the home-battery environment and score it on validation days.

Example:
    python -m battery_rl.train --algo dqn --price-only --steps 500000 --seed 0
"""

from __future__ import annotations

import os
import sys

# Unless a run asks for the GPU, hide it before torch is imported. Otherwise every CPU-only process still opens a
# ~250 MB CUDA context, and ~30 parallel runs exhaust the shared 16 GB card.
if "cuda" not in " ".join(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from .data import ROOT, load_dataset, split_days
from .env import DiscreteActions, EnvConfig, HomeBatteryEnv, Tariff
from .evaluate import evaluate, model_controller, summarize


class PeriodicEval(BaseCallback):
    """Every `every` steps, score the deterministic policy on validation days and keep the best model."""

    def __init__(self, make_env, days, every: int, out: Path, efc_limit: float | None = None, select_from: int = 0):
        """efc_limit: for constrained runs, only checkpoints whose validation cycling is within it can be 'best'.
        select_from: only checkpoints at or after this many steps can be 'best' (for PPO-Lagrangian, the step at
        which lambda starts, so an early unconstrained checkpoint that happens to cycle little is never chosen).
        Every checkpoint is also saved as ckpt_<steps>.zip so the selection rule can be revisited offline."""
        super().__init__()
        self.make_env, self.days, self.every, self.out = make_env, days, every, out
        self.efc_limit, self.select_from = efc_limit, select_from
        self.best = -np.inf
        self.history = []
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            s = summarize(evaluate(self.make_env(), self.days, model_controller(self.model)))
            s["timesteps"] = self.num_timesteps
            self.history.append(s)
            for k in ("pct_of_oracle", "median_day_pct_of_oracle", "mean_efc", "mean_bill"):
                self.logger.record(f"val/{k}", s[k])
            print(f"[{self.num_timesteps:>9}] val: {s['pct_of_oracle']:6.1f}% of oracle "
                  f"(median day {s['median_day_pct_of_oracle']:6.1f}%), EFC/day {s['mean_efc']:.2f}", flush=True)
            self.model.save(self.out / f"ckpt_{self.num_timesteps}")
            feasible = (self.efc_limit is None or s["mean_efc"] <= self.efc_limit) and self.num_timesteps >= self.select_from
            if feasible and s["pct_of_oracle"] > self.best:
                self.best = s["pct_of_oracle"]
                self.model.save(self.out / "best_model")
        return True


class LagrangianPenalty(gym.Wrapper):
    """Subtracts lambda x (cycles used this step) from the reward. lambda is set from outside by the callback.

    Cost per step is throughput / (2 x capacity), so a day's cost sum is its equivalent full cycles (EFC).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lam = 0.0

    def set_lambda(self, lam: float) -> None:
        self.lam = float(lam)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        efc_step = info["throughput_kwh"] / (2 * base.cfg.battery.capacity_kwh)
        return obs, reward - self.lam * efc_step, terminated, truncated, info


class LagrangeUpdate(BaseCallback):
    """Multiplier update after each PPO rollout, as a PID controller on the constraint violation (Stooke et al.).

        e_k = mean EFC/day over the rollout - budget
        I_k = max(0, I_{k-1} + e_k)
        lambda_k = max(0, kp * e_k + ki * I_k + kd * max(0, J_k - J_{k-1}))

    With kp = kd = 0 this is plain dual ascent with step size ki (the RCPO / PPO-Lagrangian update). The
    constraint is on the expectation over days (a CMDP), so e_k uses the mean over all days that finished during
    the rollout. lambda is in reward units per EFC (reward = reward_scale x dollars).
    """

    def __init__(self, budget: float, ki: float, kp: float = 0.0, kd: float = 0.0, start: int = 0):
        super().__init__()
        self.budget, self.ki, self.kp, self.kd, self.start = budget, ki, kp, kd, start
        self.lam, self.integral, self.prev_j = 0.0, 0.0, None
        self._efcs: list[float] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_lambda", self.lam)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "efc" in info:
                self._efcs.append(info["efc"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._efcs:
            return
        mean_efc = float(np.mean(self._efcs))
        if self.num_timesteps < self.start:  # warm-up: plain PPO first, so the policy learns arbitrage
            self.logger.record("lagrange/train_efc", mean_efc)
            self._efcs.clear()
            return
        err = mean_efc - self.budget
        self.integral = max(0.0, self.integral + err)
        deriv = 0.0 if self.prev_j is None else max(0.0, mean_efc - self.prev_j)
        self.prev_j = mean_efc
        self.lam = max(0.0, self.kp * err + self.ki * self.integral + self.kd * deriv)
        self.training_env.env_method("set_lambda", self.lam)
        self.logger.record("lagrange/lambda", self.lam)
        self.logger.record("lagrange/train_efc", mean_efc)
        self._efcs.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo", "ppo-lag", "sac"], required=True)
    ap.add_argument("--efc-budget", type=float, default=1.0, help="ppo-lag: expected cycles per day")
    ap.add_argument("--lambda-lr", type=float, default=0.1, help="ppo-lag: dual step size (the PID's ki)")
    ap.add_argument("--lambda-start", type=int, default=500_000, help="ppo-lag: steps before lambda updates begin")
    ap.add_argument("--lambda-kp", type=float, default=0.0, help="ppo-lag: proportional gain (PID Lagrangian)")
    ap.add_argument("--lambda-kd", type=float, default=0.0, help="ppo-lag: derivative gain (PID Lagrangian)")
    ap.add_argument("--log-std-init", type=float, default=0.0,
                    help="ppo: initial log std of the Gaussian policy (0 = SB3 default, std 1 = full power range)")
    ap.add_argument("--price-only", action="store_true", help="no load/solar (the proposal's first result)")
    ap.add_argument("--import-adder", type=float, default=0.0)
    ap.add_argument("--wear-cost", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--net", type=int, default=64, help="hidden width (two layers)")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--absolute-actions", action="store_true",
                    help="action x p_max then clip (original design); default is relative to the feasible range")
    ap.add_argument("--linear-actions", action="store_true", help="map [-1,1] linearly onto the feasible range")
    ap.add_argument("--oob-penalty", type=float, default=1.0, help="penalty on raw actions outside [-1, 1]")
    ap.add_argument("--battery-reward", action="store_true", help="reward = battery's effect on the bill only")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--exclude-spikes", type=float, default=None, metavar="PRICE",
                    help="stretch goal: drop training days whose max price is >= PRICE ($/kWh)")
    ap.add_argument("--buffer-size", type=int, default=1_000_000, help="sac: replay buffer size")
    ap.add_argument("--learning-starts", type=int, default=10_000, help="sac: random-action steps before updates")
    ap.add_argument("--dam-hours", type=int, default=0,
                    help="add the day-ahead price for the next N hours to the state (0 = off)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    df = load_dataset()
    split = split_days(np.sort(df["day"].unique()))
    if args.exclude_spikes is not None:
        daily_max = df.groupby("day")["price"].max()
        keep = daily_max.loc[split["train"]] < args.exclude_spikes
        print(f"excluding {int((~keep).sum())} spike days from training; {int(keep.sum())} remain")
        split["train"] = split["train"][keep.to_numpy()]
    cfg = EnvConfig(tariff=Tariff(import_adder=args.import_adder), include_load_solar=not args.price_only,
                    wear_cost=args.wear_cost, relative_actions=not args.absolute_actions,
                    linear_actions=args.linear_actions,
                    out_of_range_penalty=args.oob_penalty, battery_only_reward=args.battery_reward,
                    dam_hours=args.dam_hours)

    def make_train_env():
        env = HomeBatteryEnv(df, split["train"], cfg)
        if args.algo == "dqn":
            return DiscreteActions(env)
        return LagrangianPenalty(env) if args.algo == "ppo-lag" else env

    def make_eval_env():
        env = HomeBatteryEnv(df, split["val"], cfg)
        return DiscreteActions(env) if args.algo == "dqn" else env

    name = args.name or f"{args.algo}{'_priceonly' if args.price_only else ''}_s{args.seed}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    policy_kwargs = dict(net_arch=[args.net, args.net])
    if args.algo == "dqn":
        venv = make_vec_env(make_train_env, n_envs=1, seed=args.seed)
        model = DQN("MlpPolicy", venv, learning_rate=args.lr or 1e-4, buffer_size=300_000, learning_starts=10_000,
                    batch_size=256, gamma=args.gamma, train_freq=4, target_update_interval=4_000,
                    exploration_fraction=0.3, exploration_final_eps=0.02, policy_kwargs=policy_kwargs,
                    seed=args.seed, device=args.device, tensorboard_log=str(out / "tb"), verbose=0)
    elif args.algo == "sac":
        # One gradient step per environment step (SB3's default ratio), whatever the number of parallel envs.
        # SAC's actor outputs its own (state-dependent) log std and squashes with tanh, so --log-std-init and the
        # out-of-range penalty have no effect here.
        venv = make_vec_env(make_train_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=DummyVecEnv)
        model = SAC("MlpPolicy", venv, learning_rate=args.lr or 3e-4, buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts, batch_size=256, gamma=args.gamma, tau=0.005,
                    train_freq=1, gradient_steps=args.n_envs, ent_coef="auto", policy_kwargs=policy_kwargs,
                    seed=args.seed, device=args.device, tensorboard_log=str(out / "tb"), verbose=0)
    else:
        # The environment is cheap, so stepping all copies in one process is faster than subprocesses and lets
        # ~30 runs share the machine's 32 cores. Trajectories are identical either way for a given seed.
        venv = make_vec_env(make_train_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=DummyVecEnv)
        policy_kwargs["log_std_init"] = args.log_std_init
        model = PPO("MlpPolicy", venv, learning_rate=args.lr or 3e-4, n_steps=96 * 8, batch_size=256, n_epochs=10,
                    gamma=args.gamma, gae_lambda=0.95, ent_coef=args.ent_coef, policy_kwargs=policy_kwargs,
                    seed=args.seed, device=args.device, tensorboard_log=str(out / "tb"), verbose=0)

    efc_limit = args.efc_budget * 1.05 if args.algo == "ppo-lag" else None
    select_from = args.lambda_start if args.algo == "ppo-lag" else 0
    cb = PeriodicEval(make_eval_env, split["val"], args.eval_every, out, efc_limit, select_from)
    callbacks = [cb]
    if args.algo == "ppo-lag":
        callbacks.append(LagrangeUpdate(args.efc_budget, ki=args.lambda_lr, kp=args.lambda_kp, kd=args.lambda_kd,
                                        start=args.lambda_start))
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name="run")
    elapsed = time.time() - t0
    model.save(out / "final_model")
    (out / "val_history.json").write_text(json.dumps(cb.history, indent=2))
    if not (out / "best_model.zip").exists():  # no checkpoint met the budget: fall back to the final policy
        model.save(out / "best_model")
        print("no validation checkpoint met the cycle budget; best_model = final_model")
    print(f"done in {elapsed:.0f}s ({args.steps/elapsed:.0f} steps/s); best val {cb.best:.1f}% of oracle")


if __name__ == "__main__":
    main()
