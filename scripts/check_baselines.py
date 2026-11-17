"""Sanity checks and a first look at baseline savings.

1. Replaying the oracle's schedule through the environment must reproduce the oracle's bill exactly
   (validates that the environment and the MILP model the same battery, tariff and end-of-day rule).
2. Savings of the threshold rule and the oracle versus no battery, for the proposal's symmetric tariff
   and for a retail tariff with a delivery charge on imports.
"""

import sys
import time

import numpy as np

from battery_rl.baselines import day_arrays, run_controller, solve_oracle, threshold_controller, tune_threshold
from battery_rl.data import load_dataset, split_days
from battery_rl.env import EnvConfig, HomeBatteryEnv, Tariff

df = load_dataset()
split = split_days(np.sort(df["day"].unique()))
print({k: len(v) for k, v in split.items()})

tariffs = {
    "symmetric (proposal)": Tariff(),
    "retail: +4.5c/kWh on imports": Tariff(import_adder=0.045),
}

for name, tariff in tariffs.items():
    cfg = EnvConfig(tariff=tariff)
    env = HomeBatteryEnv(df, split["train"], cfg)

    # 1. replay check on a handful of days, including the biggest spike day
    worst = 0.0
    for d in [0, 22, 100, 198, 300]:
        price, base = day_arrays(env, d)
        orc = solve_oracle(price, base, cfg)
        sched = iter(orc.p_kw)
        info = run_controller(env, d, lambda e, o: ("kw", next(sched)))
        worst = max(worst, abs(info["bill"] - orc.bill))
        assert abs(info["final_soc"] - cfg.battery.soc0) < 1e-6, info["final_soc"]
    print(f"\n[{name}] oracle replay: max |env bill - oracle bill| = {worst:.2e} $")
    if worst > 1e-6:
        sys.exit("FAIL: environment and oracle disagree")

    # 2. baseline savings on the test days
    t0 = time.time()
    lo, hi, _ = tune_threshold(env, split["train"])
    rows = []
    for d in split["test"]:
        price, base = day_arrays(env, d)
        orc = solve_oracle(price, base, cfg)
        thr = run_controller(env, d, threshold_controller(lo, hi))
        rows.append((thr["bill_no_battery"], thr["bill"], orc.bill, thr["efc"], orc.efc))
    nb, th, oc, th_efc, oc_efc = np.array(rows).T
    print(f"threshold tuned on train: charge <= {lo*100:.2f} c/kWh, discharge >= {hi*100:.2f} c/kWh "
          f"({time.time()-t0:.0f}s)")
    print(f"test days: {len(nb)}")
    print(f"  mean daily bill   no battery {nb.mean():7.3f} $   threshold {th.mean():7.3f} $   oracle {oc.mean():7.3f} $")
    print(f"  mean daily saving threshold {np.mean(nb-th):.3f} $   oracle {np.mean(nb-oc):.3f} $"
          f"   -> threshold captures {np.sum(nb-th)/np.sum(nb-oc)*100:.1f}% of oracle")
    print(f"  equivalent full cycles/day  threshold {th_efc.mean():.2f}   oracle {oc_efc.mean():.2f}")
    print(f"  oracle saving: median day {np.median(nb-oc):.3f} $, top-3 days {np.sort(nb-oc)[-3:].round(2)} $")
