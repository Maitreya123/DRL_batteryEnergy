"""Score trained runs and the baselines on the held-out test days.

    python scripts/report.py dqn_priceonly_s{0..4}
    python scripts/report.py --group "PPO" ppo_s0 ppo_s1 ... --group "PPO-Lag" ppolag_s0 ...

For each RL method, each seed is averaged over the test days first; the 95% CI is then a t-interval over the
seed-level means (n_seeds - 1 degrees of freedom), as in the proposal. Results are reported on all test days and
on ordinary days only (no interval above SPIKE_PRICE), so a few scarcity days cannot dominate the headline.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import DQN, PPO

from battery_rl.baselines import threshold_controller, tune_threshold
from battery_rl.data import ROOT, load_dataset, split_days
from battery_rl.env import DiscreteActions, HomeBatteryEnv
from battery_rl.evaluate import evaluate, model_controller, oracle_table, run_args, run_config

SPIKE_PRICE = 1.0  # $/kWh


def pct(per_day: pd.DataFrame) -> float:
    return float(100 * (per_day.no_battery - per_day.bill).sum() / (per_day.no_battery - per_day.oracle).sum())


def scores(per_day: pd.DataFrame, ordinary: set[int]) -> dict:
    ordd = per_day[per_day.index.isin(ordinary)]
    return {"pct_all": pct(per_day), "pct_ordinary": pct(ordd), "bill": per_day.bill.mean(),
            "saving": (per_day.no_battery - per_day.bill).mean(), "efc": per_day.efc.mean()}


def ci(values: np.ndarray) -> str:
    m = values.mean()
    if len(values) < 2:
        return f"{m:7.2f}"
    h = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return f"{m:7.2f} ± {h:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--group", nargs="+", action="append", default=[], metavar=("LABEL", "RUN"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--model", default="best_model")
    args = ap.parse_args()
    groups = [(g[0], g[1:]) for g in args.group] or [("RL", args.runs)]

    df = load_dataset()
    split = split_days(np.sort(df["day"].unique()))
    days = split[args.split]
    daily_max = df.groupby("day").price.max()
    ordinary = set(int(d) for d in days if daily_max[d] < SPIKE_PRICE)
    print(f"{args.split} days: {len(days)} ({len(days) - len(ordinary)} spike days with a price above ${SPIKE_PRICE}/kWh)")

    rows = []
    cfg_cache = {}
    for label, runs in groups:
        seed_scores = []
        for run in runs:
            d = ROOT / "runs" / run
            a = run_args(run)
            cfg = run_config(a)
            cfg_cache[label] = cfg
            env = HomeBatteryEnv(df, days, cfg)
            algo = DQN if a["algo"] == "dqn" else PPO
            if a["algo"] == "dqn":
                env = DiscreteActions(env)
            model = algo.load(d / args.model, device="cpu")
            seed_scores.append(scores(evaluate(env, days, model_controller(model)), ordinary))
        s = pd.DataFrame(seed_scores)
        rows.append((label, len(runs), {k: ci(s[k].to_numpy()) for k in s.columns}))

    # deterministic baselines, with the configuration of the first group (the main experiment)
    cfg = list(cfg_cache.values())[0]
    env = HomeBatteryEnv(df, split["train"], cfg)
    lo, hi, _ = tune_threshold(env, split["train"])
    test_env = HomeBatteryEnv(df, days, cfg)
    thr = evaluate(test_env, days, threshold_controller(lo, hi))
    rows.append(("Threshold rule", 1, {k: f"{v:7.2f}" for k, v in scores(thr, ordinary).items()}))
    orc = thr.assign(bill=thr.oracle, efc=thr.oracle_efc)
    rows.append(("Oracle (hindsight)", 1, {k: f"{v:7.2f}" for k, v in scores(orc, ordinary).items()}))
    for cap in (1.0, 0.75, 0.5):
        capped = oracle_table(test_env, days, efc_cap=cap)
        orc_c = thr.assign(bill=capped["bill"].to_numpy(), efc=capped["efc"].to_numpy())
        rows.append((f"Oracle, <= {cap} EFC/day", 1, {k: f"{v:7.2f}" for k, v in scores(orc_c, ordinary).items()}))
    nob = thr.assign(bill=thr.no_battery, efc=0.0)
    rows.append(("No battery", 1, {k: f"{v:7.2f}" for k, v in scores(nob, ordinary).items()}))

    head = f"{'method':<24}{'seeds':>6}  {'% oracle (all)':>16}  {'% oracle (ordinary)':>19}  {'bill $/day':>16}  {'saving $/day':>16}  {'EFC/day':>16}"
    print(head)
    print("-" * len(head))
    for label, n, r in rows:
        print(f"{label:<24}{n:>6}  {r['pct_all']:>16}  {r['pct_ordinary']:>19}  {r['bill']:>16}  {r['saving']:>16}  {r['efc']:>16}")


if __name__ == "__main__":
    main()
