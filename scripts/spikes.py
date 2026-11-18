"""Stretch goal: how well do agents handle ERCOT scarcity (price-spike) days?

Spike day = a day whose highest 15-minute price is at least SPIKE_PRICE. Each group is scored on:
  - "test ordinary": test days without a spike (reference),
  - "test spikes":   the 4 spike days in the test split (unseen by every model),
  - "all spikes":    all 17 spike days of 2018 (only meaningful for models trained without spike days, which
                     have seen none of them; models trained on all training days saw 13 of the 17).
The score is the share of the oracle's savings, summed over the days in the set (seed-level, then a t-interval
over seeds, as in report.py). Per-day oracle shares are also printed for the 4 test spike days.

    python scripts/spikes.py --group PPO ppo_s0 ... --group "PPO (no spikes)" noSpike_ppo_s0 ...
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from battery_rl.data import ROOT, load_dataset, split_days
from battery_rl.env import HomeBatteryEnv
from battery_rl.evaluate import evaluate, model_controller, run_args, run_config

from report import SPIKE_PRICE, ci, pct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("LABEL", "RUN"))
    args = ap.parse_args()

    df = load_dataset()
    split = split_days(np.sort(df["day"].unique()))
    daily_max = df.groupby("day")["price"].max()
    spikes = np.sort(daily_max[daily_max >= SPIKE_PRICE].index.to_numpy())
    test_spikes = np.intersect1d(spikes, split["test"])
    test_ordinary = np.setdiff1d(split["test"], spikes)
    days = np.union1d(spikes, split["test"])
    sets = {"test ordinary": test_ordinary, "test spikes": test_spikes, "all spikes": spikes}
    print(f"spike days (max price >= ${SPIKE_PRICE}/kWh): {len(spikes)}; in test: {[int(d) for d in test_spikes]}")

    header = f"{'method':<24}" + "".join(f"{k + ' (%)':>20}" for k in sets) + "   per test-spike day (%)"
    print(header)
    print("-" * len(header))
    for label, runs in [(g[0], g[1:]) for g in args.group]:
        per_set = {k: [] for k in sets}
        per_day = []
        for run in runs:
            cfg = run_config(run_args(run))
            model = PPO.load(ROOT / "runs" / run / "best_model", device="cpu")
            res = evaluate(HomeBatteryEnv(df, days, cfg), days, model_controller(model))
            for k, ds in sets.items():
                per_set[k].append(pct(res.loc[ds]))
            sp = res.loc[test_spikes]
            per_day.append(100 * (sp.no_battery - sp.bill) / (sp.no_battery - sp.oracle))
        row = f"{label:<24}" + "".join(f"{ci(np.array(v)):>20}" for v in per_set.values())
        row += "   " + " ".join(f"d{d}:{v:5.1f}" for d, v in pd.concat(per_day, axis=1).mean(axis=1).items())
        print(row)


if __name__ == "__main__":
    main()
