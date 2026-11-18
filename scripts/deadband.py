"""Where do PPO-Lagrangian's cycles come from, and does a deadband on small flows reach the 0.5 budget?

Produces the two analyses behind the "~0.55 cycles/day floor" section of the README, on the 91 test days:

1. results/cycle_floor_small_flows.txt: per run, cycles/day, the share of idle steps (|p| < 0.05 kW), and the
   cycles/day contributed by steps in each |p| band (kW).
2. results/cycle_floor_deadband.txt: per run and deadband, the share of oracle savings and cycles/day when any
   requested |p| below the deadband is replaced by 0 kW at evaluation time (no retraining).

    python scripts/deadband.py
    python scripts/deadband.py --runs ppolag_b0.5_s0 ... --deadbands 0 0.5 1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from battery_rl.data import ROOT, load_dataset, split_days
from battery_rl.env import HomeBatteryEnv
from battery_rl.evaluate import oracle_table, run_args, run_config

DEFAULT_RUNS = [f"ppolag_b0.5_s{s}" for s in range(3)] + [f"ppolag_b0.5_pid_s{s}" for s in range(3)]
BANDS = [0, 0.25, 0.5, 1, 2, 5.01]  # kW


def rollout(env: HomeBatteryEnv, model, day: int, deadband: float = 0.0) -> tuple[list[float], dict]:
    """One day with the deterministic policy; requested |p| < deadband becomes 0 kW. Returns |p| per step and info."""
    obs, _ = env.reset(options={"day": day})
    flows, done = [], False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        p = env.action_to_kw(float(np.clip(a[0], -1, 1)))
        if abs(p) < deadband:
            p = 0.0
        obs, _, done, _, info = env.step_kw(p)  # step_kw applies the same feasibility clip as step()
        flows.append(abs(info["p_kw"]))
    return flows, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--deadbands", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    args = ap.parse_args()

    df = load_dataset()
    days = split_days(np.sort(df["day"].unique()))["test"]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    flow_lines = ["cycles/day by battery power band, deterministic policy, test days; idle = |p| < 0.05 kW",
                  f"{'run':22s} {'EFC/day':>8s} {'idle':>5s} {'EFC |p|<1kW':>12s}  EFC by |p| band "
                  + " ".join(f"[{lo:g},{hi:g})" for lo, hi in zip(BANDS[:-1], BANDS[1:]))]
    db_lines = ["deadband: requested |p| below the threshold is replaced by 0 kW at evaluation time (no retraining)",
                f"{'run':22s} {'deadband kW':>11s} {'% oracle':>9s} {'EFC/day':>8s}"]
    for run in args.runs:
        cfg = run_config(run_args(run))
        env = HomeBatteryEnv(df, days, cfg)
        model = PPO.load(ROOT / "runs" / run / "best_model", device="cpu")
        b = cfg.battery

        p = np.concatenate([rollout(env, model, int(d))[0] for d in days])
        efc = p * b.dt_h / (2 * b.capacity_kwh) / len(days)
        by_band = pd.Series(efc).groupby(pd.cut(p, BANDS, right=False), observed=False).sum()
        flow_lines.append(f"{run:22s} {efc.sum():8.2f} {np.mean(p < 0.05):5.0%} {by_band.iloc[:3].sum():12.2f}  "
                          + " ".join(f"{v:.3f}" for v in by_band))

        orc = oracle_table(env, days)
        for db in args.deadbands:
            sav = osav = cyc = 0.0
            for d in days:
                _, info = rollout(env, model, int(d), db)
                sav += info["bill_no_battery"] - info["bill"]
                osav += info["bill_no_battery"] - orc.loc[int(d), "bill"]
                cyc += info["efc"]
            db_lines.append(f"{run:22s} {db:11.1f} {100 * sav / osav:9.1f} {cyc / len(days):8.2f}")

    (out / "cycle_floor_small_flows.txt").write_text("\n".join(flow_lines) + "\n")
    (out / "cycle_floor_deadband.txt").write_text("\n".join(db_lines) + "\n")
    print("\n".join(flow_lines + [""] + db_lines))


if __name__ == "__main__":
    main()
