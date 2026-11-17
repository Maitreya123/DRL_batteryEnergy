"""Animated replay of one test day, for the presentation video (MP4, 1280x720).

    python scripts/animate.py --day 332 --run ppo_s4      # Nov 29: an ordinary day with an evening peak
    python scripts/animate.py --day 22  --run ppo_s4      # Jan 23: the $5.79/kWh scarcity day

Panels: ERCOT price with a moving cursor; state of charge for PPO and the hindsight oracle, with battery gauges;
cumulative electricity cost for no battery, the threshold rule, PPO and the oracle. Bills include the household's
own load minus solar, so the gaps between the cost lines are exactly what each battery controller saves.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO

from battery_rl.baselines import day_arrays, solve_oracle, threshold_controller, tune_threshold
from battery_rl.data import LST, ROOT, load_dataset, split_days
from battery_rl.env import STEPS_PER_DAY, EnvConfig, HomeBatteryEnv
from battery_rl.evaluate import run_args, run_config

# Reference palette (dataviz skill, light mode), matching scripts/plots.py.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "lines.linewidth": 2.5,
})


def rollout(env: HomeBatteryEnv, day: int, act):
    """Run one day; act(env, obs) -> ('kw', value) or ('action', array). Returns per-step cost and SoC."""
    obs, _ = env.reset(options={"day": day})
    costs, socs, done = [], [env.soc], False
    while not done:
        kind, val = act(env, obs)
        obs, _, done, _, info = env.step_kw(val) if kind == "kw" else env.step(val)
        costs.append(info["cost"])
        socs.append(info["soc"])
    return np.array(costs), np.array(socs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--run", default="ppo_s4")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the final frame")
    args = ap.parse_args()

    df = load_dataset()
    split = split_days(np.sort(df["day"].unique()))
    cfg = run_config(run_args(args.run))
    env = HomeBatteryEnv(df, [args.day], cfg)
    b = cfg.battery

    # threshold thresholds are tuned on training days exactly as in report.py
    lo, hi, _ = tune_threshold(HomeBatteryEnv(df, split["train"], EnvConfig()), split["train"])
    model = PPO.load(ROOT / "runs" / args.run / "best_model", device="cpu")

    c_none, _ = rollout(env, args.day, lambda e, o: ("kw", 0.0))
    c_thr, _ = rollout(env, args.day, threshold_controller(lo, hi))
    c_ppo, s_ppo = rollout(env, args.day, lambda e, o: ("action", model.predict(o, deterministic=True)[0]))
    price, base = day_arrays(env, args.day)
    orc = solve_oracle(price, base, cfg)
    c_orc = cfg.tariff.cost(price, (base + orc.p_kw) * b.dt_h)
    s_orc = [b.soc0]
    for p in orc.p_kw:
        s_orc.append(b.soc_after(s_orc[-1], p))
    s_orc = np.array(s_orc)
    assert abs(c_none.sum() - np.sum(cfg.tariff.cost(price, base * b.dt_h))) < 1e-9

    hours = np.arange(STEPS_PER_DAY + 1) / 4
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
           {"none": c_none, "thr": c_thr, "ppo": c_ppo, "orc": c_orc}.items()}
    pct_ppo, pct_orc = (s_ppo / b.capacity_kwh * 100, s_orc / b.capacity_kwh * 100)
    cents = price * 100
    date = pd.Timestamp("2018-01-01", tz=LST) + pd.Timedelta(days=args.day)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[6, 1], height_ratios=[1, 1.15, 1.15], hspace=0.45, wspace=0.08,
                          left=0.07, right=0.97, top=0.88, bottom=0.08)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_p)
    ax_g = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_p)
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis("off")
    fig.suptitle(f"One test day, replayed: {date:%A, %B %-d, 2018}", fontsize=16, fontweight="bold", color=INK,
                 x=0.07, ha="left")
    clock = fig.text(0.97, 0.935, "", ha="right", fontsize=15, color=INK, family="monospace")

    # price: whole day faint, revealed part solid
    ax_p.step(hours[:-1], cents, where="post", color=GRID, lw=2)
    (p_line,) = ax_p.step([], [], where="post", color=INK2, lw=2)
    ax_p.set_ylabel("Price (¢/kWh)")
    ax_p.set_title("ERCOT real-time price", loc="left")
    ax_p.set_xlim(0, 24)
    ax_p.set_ylim(min(0, cents.min()) - 1, cents.max() * 1.12 + 1)
    cursors = [ax.axvline(0, color=MUTED, lw=1.2, ls=":") for ax in (ax_p, ax_s, ax_c)]

    # state of charge
    (l_orc,) = ax_s.plot([], [], color=ORANGE, label="Oracle (knows the whole day)")
    (l_ppo,) = ax_s.plot([], [], color=BLUE, label="PPO agent")
    ax_s.set_ylim(-3, 103)
    ax_s.set_ylabel("Battery charge (%)")
    ax_s.set_title("Battery state of charge", loc="left")
    ax_s.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=11, ncol=2)

    # gauges
    ax_g.set_xlim(0, 2)
    ax_g.set_ylim(0, 100)
    ax_g.axis("off")
    gauges = {}
    for x, name, color in ((0.15, "PPO", BLUE), (1.15, "Oracle", ORANGE)):
        ax_g.add_patch(Rectangle((x, 0), 0.7, 100, fill=False, edgecolor=INK2, lw=1.5))
        bar = ax_g.add_patch(Rectangle((x, 0), 0.7, 50, color=color, alpha=0.85))
        ax_g.text(x + 0.35, -6, name, ha="center", va="top", fontsize=11, color=INK2)
        arrow = ax_g.text(x + 0.35, 104, "", ha="center", va="bottom", fontsize=16, color=INK2)
        gauges[name] = (bar, arrow)

    # cumulative cost
    series = [("none", "No battery", MUTED, "-"), ("thr", "Threshold rule", INK2, "--"),
              ("ppo", "PPO agent", BLUE, "-"), ("orc", "Oracle", ORANGE, "-")]
    c_lines = {}
    for key, label, color, ls in series:
        (c_lines[key],) = ax_c.plot([], [], color=color, ls=ls, label=label)
    allc = np.concatenate(list(cum.values()))
    pad = 0.08 * (allc.max() - allc.min() + 1e-9)
    ax_c.set_ylim(allc.min() - pad, allc.max() + pad)
    ax_c.set_ylabel("Cost so far ($)")
    ax_c.set_xlabel("Hour of day (local standard time)")
    ax_c.set_title("Electricity cost so far today, household + battery (below 0 = net income)", loc="left")
    ax_c.set_xticks(range(0, 25, 3))
    ax_c.legend(loc="lower left", frameon=False, fontsize=10, ncol=4)
    info = ax_info.text(0.08, 1.05, "", va="top", fontsize=11, color=INK, linespacing=1.35, family="monospace")

    def draw(i: int):
        t = hours[: i + 1]
        p_line.set_data(hours[: max(i, 1)], cents[: max(i, 1)])
        for c in cursors:
            c.set_xdata([hours[i]])
        l_ppo.set_data(t, pct_ppo[: i + 1])
        l_orc.set_data(t, pct_orc[: i + 1])
        for name, pct, flows in (("PPO", pct_ppo, np.diff(s_ppo)), ("Oracle", pct_orc, np.diff(s_orc))):
            bar, arrow = gauges[name]
            bar.set_height(pct[i])
            f = flows[i - 1] if i > 0 else 0.0
            arrow.set_text("▲" if f > 1e-3 else ("▼" if f < -1e-3 else "–"))
        for key, *_ in series:
            c_lines[key].set_data(t, cum[key][: i + 1])
        hh, mm = divmod(i * 15, 60)
        clock.set_text(f"{hh:02d}:{mm:02d}")
        if i == STEPS_PER_DAY:  # final frame: direct labels and the day's result
            saved_ppo, saved_orc = cum["none"][-1] - cum["ppo"][-1], cum["none"][-1] - cum["orc"][-1]
            saved_thr = cum["none"][-1] - cum["thr"][-1]
            share = f"{100 * saved_ppo / saved_orc:.0f}%" if saved_orc > 1e-6 else "n/a"
            info.set_text(f"Saved vs.\nno battery\n\nPPO    ${saved_ppo:5.2f}\nRule   ${saved_thr:5.2f}\n"
                          f"Oracle ${saved_orc:5.2f}\n\nPPO: {share}\nof oracle")
        return []

    frames = list(range(STEPS_PER_DAY + 1)) + [STEPS_PER_DAY] * int(args.hold * args.fps)
    out = ROOT / "results" / "videos" / f"day{args.day}_{date:%b%d}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=100):
        for i in frames:
            draw(i)
            writer.grab_frame()
    fig.savefig(out.with_suffix(".png"), dpi=100)  # the final frame as a still, for checking and for slides
    print(f"wrote {out} ({len(frames) / args.fps:.0f}s); saved: PPO ${cum['none'][-1]-cum['ppo'][-1]:.2f}, "
          f"rule ${cum['none'][-1]-cum['thr'][-1]:.2f}, oracle ${cum['none'][-1]-cum['orc'][-1]:.2f}")


if __name__ == "__main__":
    main()
