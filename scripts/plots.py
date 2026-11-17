"""Figures for the report (PNG, 200 dpi) in results/figures/.

1. tradeoff.png       savings (% of oracle, test days) vs. cycles/day, per seed and per-method mean, with the
                      oracle's budget frontier and the threshold rule for reference.
2. learning.png       validation % of oracle during training, PPO and PPO-Lagrangian (budget 0.75), per seed.
3. example_day.png    one test day: price (top) and battery charge for PPO vs. the oracle (bottom).

Per-run test scores are cached in results/per_run_test.csv; delete it to recompute.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from battery_rl.baselines import day_arrays, run_controller, solve_oracle, threshold_controller, tune_threshold
from battery_rl.data import LST, ROOT, load_dataset, split_days
from battery_rl.env import EnvConfig, HomeBatteryEnv
from battery_rl.evaluate import evaluate, model_controller, oracle_table

OUT = ROOT / "results"
FIG = OUT / "figures"

# Reference palette (dataviz skill, light mode): categorical slots 1-3 in fixed order, neutral ink for references.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
})

GROUPS = {
    "PPO": [f"ppo_s{s}" for s in range(5)],
    **{f"PPO-Lag b{b}": [f"ppolag_b{b}_s{s}" for s in range(5)] for b in ("1.0", "0.75", "0.5")},
    **{f"PPO wear {w}": [f"ppowear_w{w}_s{s}" for s in range(3)] for w in ("0.005", "0.01", "0.02", "0.04")},
}


def cfg_for(run: str) -> EnvConfig:
    a = json.loads((ROOT / "runs" / run / "args.json").read_text())
    return EnvConfig(relative_actions=not a.get("absolute_actions", "oob_penalty" not in a),
                     linear_actions=a.get("linear_actions", False),
                     battery_only_reward=a.get("battery_reward", False))


def per_run_scores(df, days) -> pd.DataFrame:
    path = OUT / "per_run_test.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for group, runs in GROUPS.items():
        for run in runs:
            model = PPO.load(ROOT / "runs" / run / "best_model", device="cpu")
            r = evaluate(HomeBatteryEnv(df, days, cfg_for(run)), days, model_controller(model))
            pct = 100 * (r.no_battery - r.bill).sum() / (r.no_battery - r.oracle).sum()
            rows.append(dict(group=group, run=run, pct=pct, efc=r.efc.mean()))
    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    return out


def fig_tradeoff(df, split, scores):
    days = split["test"]
    env = HomeBatteryEnv(df, days, EnvConfig())
    free = oracle_table(env, days)
    thr_lo, thr_hi, _ = tune_threshold(HomeBatteryEnv(df, split["train"], EnvConfig()), split["train"])
    thr = evaluate(env, days, threshold_controller(thr_lo, thr_hi))
    nb = thr.no_battery
    oracle_saving = (nb - free["bill"].to_numpy()).sum()

    frontier = [(0.0, 0.0)]
    for cap in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        t = oracle_table(env, days, efc_cap=cap)
        frontier.append((t.efc.mean(), 100 * (nb - t["bill"].to_numpy()).sum() / oracle_saving))
    frontier.append((free.efc.mean(), 100.0))
    fx, fy = zip(*sorted(frontier))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(fx, fy, color=MUTED, lw=2, ls="--", zorder=1)
    ax.annotate("Oracle with a cycle cap", (fx[3], fy[3]), xytext=(8, 8), textcoords="offset points",
                color=INK2, fontsize=9)
    t_pct = 100 * (thr.no_battery - thr.bill).sum() / oracle_saving
    ax.scatter([thr.efc.mean()], [t_pct], marker="D", s=60, color=MUTED, edgecolor=SURFACE, lw=2, zorder=3)
    ax.annotate("Threshold rule", (thr.efc.mean(), t_pct), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", color=INK2, fontsize=9)

    series = [("PPO-Lagrangian (cycle budget)", "PPO-Lag", BLUE, "o"),
              ("PPO + fixed wear cost", "PPO wear", ORANGE, "s"),
              ("PPO (unconstrained)", "PPO", AQUA, "^")]
    for label, prefix, color, marker in series:
        sel = scores[scores.group.str.startswith(prefix)] if prefix != "PPO" else scores[scores.group == "PPO"]
        ax.scatter(sel.efc, sel.pct, s=18, color=color, alpha=0.35, marker=marker, lw=0, zorder=2)
        means = sel.groupby("group")[["efc", "pct"]].mean()
        ax.scatter(means.efc, means.pct, s=80, color=color, marker=marker, edgecolor=SURFACE, lw=2,
                   zorder=4, label=label)
        # Budget labels sit above their marks, wear-cost labels below, so the two series never collide.
        for g, r in means.iterrows():
            tag = g.split()[-1]
            if prefix == "PPO-Lag":
                b = tag.lstrip("b")
                dx = {"0.5": -8, "0.75": 0, "1.0": 8}.get(b, 0)
                ha = {"0.5": "right", "0.75": "center", "1.0": "left"}.get(b, "center")
                ax.annotate(f"≤{b}", (r.efc, r.pct), xytext=(dx, 11), textcoords="offset points",
                            ha=ha, color=INK2, fontsize=8)
            elif prefix == "PPO wear":
                ax.annotate(f"{float(tag) * 100:g}¢", (r.efc, r.pct), xytext=(0, -16), textcoords="offset points",
                            ha="center", color=INK2, fontsize=8)

    ax.set_xlabel("Equivalent full cycles per day (test days)")
    ax.set_ylabel("Savings, % of unconstrained oracle")
    ax.set_title("Savings vs. battery cycling on held-out days")
    ax.set_xlim(0, 1.85)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, 0.01, "Small marks: individual seeds. Large marks: mean over seeds. Labels: cycle budget (≤) or wear cost (¢/kWh).",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "tradeoff.png", dpi=200)
    plt.close(fig)


def fig_learning():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, runs) in zip(axes, [("PPO", GROUPS["PPO"]), ("PPO-Lagrangian, budget 0.75", GROUPS["PPO-Lag b0.75"])]):
        for i, run in enumerate(runs):
            h = pd.DataFrame(json.loads((ROOT / "runs" / run / "val_history.json").read_text()))
            ax.plot(h.timesteps / 1e6, h.pct_of_oracle, color=BLUE, lw=1.5, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Training steps (millions)")
        ax.axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel("Validation savings, % of oracle")
    fig.text(0.01, 0.01, "One line per seed (5 seeds).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "learning.png", dpi=200)
    plt.close(fig)


def fig_example_day(df, split, scores):
    # the median-scoring PPO seed, on the test day with the largest oracle saving among ordinary days
    ppo = scores[scores.group == "PPO"].sort_values("pct")
    run = ppo.iloc[len(ppo) // 2].run
    cfg = cfg_for(run)
    env = HomeBatteryEnv(df, split["test"], cfg)
    model = PPO.load(ROOT / "runs" / run / "best_model", device="cpu")
    daily_max = df.groupby("day").price.max()
    cands = [d for d in split["test"] if daily_max[d] < 1.0]
    orc = oracle_table(env, cands)
    nb = {d: run_controller(env, d, lambda e, o: ("kw", 0.0))["bill"] for d in cands}
    day = max(cands, key=lambda d: nb[d] - orc.loc[d, "bill"])

    price, base = day_arrays(env, day)
    o = solve_oracle(price, base, env.cfg)
    soc_o = [env.cfg.battery.soc0]
    for p in o.p_kw:
        soc_o.append(env.cfg.battery.soc_after(soc_o[-1], p))
    soc_r = [env.cfg.battery.soc0]
    obs, _ = env.reset(options={"day": day})
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(a)
        soc_r.append(info["soc"])
    t = np.arange(97) / 4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    ax1.step(np.arange(96) / 4, price * 100, where="post", color=INK2, lw=1.5)
    ax1.set_ylabel("Price (¢/kWh)")
    date = pd.Timestamp("2018-01-01", tz=LST) + pd.Timedelta(days=int(day))
    ax1.set_title(f"Example test day ({date:%b %d, 2018}): ERCOT price and battery charge")
    ax2.plot(t, np.array(soc_o) / env.cfg.battery.capacity_kwh * 100, color=ORANGE, label="Oracle (hindsight)")
    ax2.plot(t, np.array(soc_r) / env.cfg.battery.capacity_kwh * 100, color=BLUE, label="PPO")
    ax2.set_ylabel("State of charge (%)")
    ax2.set_xlabel("Hour of day (local standard time)")
    ax2.set_ylim(-3, 103)
    ax2.set_xticks(range(0, 25, 3))
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "example_day.png", dpi=200)
    plt.close(fig)
    return day, run


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    split = split_days(np.sort(df["day"].unique()))
    scores = per_run_scores(df, split["test"])
    print(scores.groupby("group")[["pct", "efc"]].agg(["mean", "std"]).round(2).to_string())
    fig_tradeoff(df, split, scores)
    fig_learning()
    day, run = fig_example_day(df, split, scores)
    print(f"figures written to {FIG} (example day {day}, run {run})")


if __name__ == "__main__":
    main()
