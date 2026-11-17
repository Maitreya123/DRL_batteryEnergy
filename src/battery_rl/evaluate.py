"""Score controllers on a set of days against the no-battery bill and the perfect-foresight oracle."""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import day_arrays, run_controller, solve_oracle
from .data import PROCESSED
from .env import EnvConfig, HomeBatteryEnv


def _cfg_key(cfg: EnvConfig, efc_cap) -> str:
    blob = json.dumps(
        dict(b=cfg.battery.__dict__, t=cfg.tariff.__dict__, ls=cfg.include_load_solar, w=cfg.wear_cost, cap=efc_cap),
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def oracle_table(env: HomeBatteryEnv, days, efc_cap: float | None = None) -> pd.DataFrame:
    """Oracle bill and cycles per day, cached on disk per environment configuration."""
    env = env.unwrapped
    path = PROCESSED / f"oracle_{_cfg_key(env.cfg, efc_cap)}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["bill", "efc"])
    todo = [int(d) for d in days if int(d) not in have.index]
    if todo:
        rows = {}
        for d in todo:
            price, base = day_arrays(env, d)
            r = solve_oracle(price, base, env.cfg, efc_cap)
            rows[d] = {"bill": r.bill, "efc": r.efc}
        have = pd.concat([have, pd.DataFrame.from_dict(rows, orient="index")]).sort_index()
        have.index = have.index.astype(int)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        have.to_parquet(tmp)
        os.replace(tmp, path)  # atomic, so parallel runs never read a half-written file
    return have.loc[[int(d) for d in days]]


def model_controller(model, deterministic: bool = True):
    def ctrl(env, obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return "action", action

    return ctrl


def evaluate(env: HomeBatteryEnv, days, controller, efc_cap: float | None = None) -> pd.DataFrame:
    """Per-day bill, no-battery bill, oracle bill and cycles. `controller` follows baselines.run_controller."""
    orc = oracle_table(env, days, efc_cap)
    rows = []
    for d in days:
        info = run_controller(env, int(d), controller)
        rows.append(dict(day=int(d), bill=info["bill"], no_battery=info["bill_no_battery"], efc=info["efc"],
                         oracle=orc.loc[int(d), "bill"], oracle_efc=orc.loc[int(d), "efc"]))
    return pd.DataFrame(rows).set_index("day")


def summarize(per_day: pd.DataFrame) -> dict:
    saving = per_day["no_battery"] - per_day["bill"]
    oracle_saving = per_day["no_battery"] - per_day["oracle"]
    return {
        "days": len(per_day),
        "mean_bill": float(per_day["bill"].mean()),
        "mean_saving": float(saving.mean()),
        "pct_of_oracle": float(100 * saving.sum() / oracle_saving.sum()),
        # same ratio with each day weighted equally, so a few spike days cannot dominate it
        "median_day_pct_of_oracle": float(100 * np.median(saving / oracle_saving.where(oracle_saving > 1e-9))),
        "mean_efc": float(per_day["efc"].mean()),
    }
