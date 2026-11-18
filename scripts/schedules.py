"""Why does PPO only tie the threshold rule on ordinary days? Hour-by-hour comparison of schedules.

For every ordinary test day (no price above SPIKE_PRICE) and every controller (PPO seeds, the tuned threshold rule,
the hindsight oracle), record the battery power, state of charge and the money the battery saved in each step.
Then average by hour of day:

    results/schedules_by_hour.csv   mean power (kW), charge (%), saving ($) per hour and controller
    results/figures/schedules.png   price, power and cumulative saving through an average ordinary day
    stdout                          where PPO gains and loses against the rule, and how often it is idle

    python scripts/schedules.py --runs ppo_s0 ppo_s1 ppo_s2 ppo_s3 ppo_s4
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from battery_rl.baselines import day_arrays, solve_oracle, threshold_controller, tune_threshold
from battery_rl.data import ROOT, load_dataset, split_days
from battery_rl.env import STEPS_PER_DAY, EnvConfig, HomeBatteryEnv
from battery_rl.evaluate import run_args, run_config

from report import SPIKE_PRICE

BLUE, ORANGE, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def rollout(env: HomeBatteryEnv, day: int, act) -> pd.DataFrame:
    """Per-step battery power, charge after the step, and the battery's saving (no-battery cost - cost)."""
    obs, _ = env.reset(options={"day": day})
    rows, done = [], False
    while not done:
        kind, val = act(env, obs)
        i = env._idx()
        base = env.base_load(i)
        cost0 = float(env.cfg.tariff.cost(env.price[i], base * env.cfg.battery.dt_h))
        obs, _, done, _, info = env.step_kw(val) if kind == "kw" else env.step(val)
        rows.append((info["p_kw"], info["soc"], cost0 - info["cost"]))
    out = pd.DataFrame(rows, columns=["p_kw", "soc", "saving"])
    out["step"] = np.arange(STEPS_PER_DAY)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    df = load_dataset()
    split = split_days(np.sort(df["day"].unique()))
    daily_max = df.groupby("day")["price"].max()
    days = [int(d) for d in split["test"] if daily_max[d] < SPIKE_PRICE]
    cfg = run_config(run_args(args.runs[0]))
    env = HomeBatteryEnv(df, days, cfg)
    b = cfg.battery

    lo, hi, _ = tune_threshold(HomeBatteryEnv(df, split["train"], EnvConfig()), split["train"])
    frames = []
    for d in days:
        price, base = day_arrays(env, d)
        orc = iter(solve_oracle(price, base, cfg).p_kw)
        for name, act in [("Threshold rule", threshold_controller(lo, hi)), ("Oracle", lambda e, o: ("kw", next(orc)))]:
            frames.append(rollout(env, d, act).assign(controller=name, day=d, price=price))
    for run in args.runs:
        model = PPO.load(ROOT / "runs" / run / "best_model", device="cpu")
        act = lambda e, o, m=model: ("action", m.predict(o, deterministic=True)[0])  # noqa: E731
        for d in days:
            price, _ = day_arrays(env, d)
            frames.append(rollout(env, d, act).assign(controller="PPO", seed=run, day=d, price=price))
    steps = pd.concat(frames, ignore_index=True)
    steps["hour"] = steps["step"] // 4
    steps["soc_pct"] = 100 * steps["soc"] / b.capacity_kwh

    # PPO is averaged over seeds as well as days, so each controller has one profile.
    by_hour = (steps.groupby(["controller", "hour"])
               .agg(price_c=("price", lambda x: 100 * x.mean()), p_kw=("p_kw", "mean"),
                    soc_pct=("soc_pct", "mean"), saving=("saving", lambda x: x.sum()))
               .reset_index())
    n_days = {c: steps[steps.controller == c].groupby(["day", "seed"] if c == "PPO" else ["day"]).ngroups
              for c in steps.controller.unique()}
    by_hour["saving"] /= by_hour["controller"].map(n_days)  # $ per day, by hour
    (ROOT / "results").mkdir(exist_ok=True)
    by_hour.to_csv(ROOT / "results" / "schedules_by_hour.csv", index=False, float_format="%.5f")

    wide = by_hour.pivot(index="hour", columns="controller", values="saving")
    total = wide.sum()
    print(f"{len(days)} ordinary test days, PPO seeds: {', '.join(args.runs)}; rule: charge <= {100*lo:.2f}c, "
          f"discharge >= {100*hi:.2f}c")
    print("mean saving per day ($):", {k: round(v, 3) for k, v in total.items()})
    print(f"share of oracle: PPO {100*total['PPO']/total['Oracle']:.1f}%, "
          f"rule {100*total['Threshold rule']/total['Oracle']:.1f}%")
    diff = (wide["PPO"] - wide["Threshold rule"])
    print("\nPPO minus rule, saving by hour ($/day):")
    print(diff.round(4).to_string())
    print(f"\nhours where PPO gains: {diff[diff > 0].sum():+.3f} $/day; where it loses: {diff[diff < 0].sum():+.3f} $/day")

    s = steps.assign(idle=steps.p_kw.abs() < 0.05, full=steps.p_kw.abs() > 0.95 * b.p_max_kw)
    print("\nshare of steps idle / at full power:")
    print(s.groupby("controller")[["idle", "full"]].mean().round(3).to_string())
    cyc = steps.groupby(["controller", "day", "seed"], dropna=False).p_kw.apply(
        lambda p: p.abs().sum() * b.dt_h / (2 * b.capacity_kwh)).groupby("controller").mean()
    print("\ncycles/day:", cyc.round(2).to_dict())
    # value per cycle: how much each controller earns for each equivalent full cycle
    print("saving per cycle ($):", (total / cyc).round(3).to_dict())

    # price spread captured: average price paid when charging vs received when discharging
    def spread(g):
        ch, dis = g[g.p_kw > 0.05], g[g.p_kw < -0.05]
        return pd.Series({"charge_price_c": 100 * np.average(ch.price, weights=ch.p_kw),
                          "discharge_price_c": 100 * np.average(dis.price, weights=-dis.p_kw)})
    print("\nenergy-weighted price when charging / discharging (c/kWh):")
    print(steps.groupby("controller").apply(spread, include_groups=False).round(2).to_string())

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
    ax = axes[0]
    ax.plot(wide.index + 0.5, by_hour[by_hour.controller == "Oracle"].set_index("hour").price_c, color=INK2)
    ax.set_ylabel("Mean price (¢/kWh)")
    ax.set_title(f"An average ordinary test day ({len(days)} days)")
    colors = {"PPO": BLUE, "Threshold rule": MUTED, "Oracle": ORANGE}
    styles = {"PPO": "-", "Threshold rule": "--", "Oracle": "-"}
    for c in ("Oracle", "Threshold rule", "PPO"):
        h = by_hour[by_hour.controller == c].set_index("hour")
        axes[1].plot(h.index + 0.5, h.p_kw, color=colors[c], ls=styles[c], label=c)
        axes[2].plot(np.r_[0, h.index + 1], np.r_[0, h.saving.cumsum()], color=colors[c], ls=styles[c], label=c)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_ylabel("Mean battery power (kW)\n(+ charge, − discharge)")
    axes[1].legend(loc="upper left", fontsize=9, ncol=3)
    axes[2].set_ylabel("Cumulative saving ($)")
    axes[2].set_xlabel("Hour of day (local standard time)")
    axes[2].set_xticks(range(0, 25, 3))
    fig.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "schedules.png", dpi=200)
    print("\nwrote results/schedules_by_hour.csv and results/figures/schedules.png")


if __name__ == "__main__":
    main()
