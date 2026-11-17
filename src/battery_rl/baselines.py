"""Baselines: no battery, a tuned price-threshold rule, and the perfect-foresight MILP oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .env import STEPS_PER_DAY, HomeBatteryEnv


@dataclass
class OracleResult:
    bill: float
    p_kw: np.ndarray  # grid-side battery power per step
    efc: float


def solve_oracle(price, base_kw, env_cfg, efc_cap: float | None = None) -> OracleResult:
    """Minimum-cost schedule for one day with full knowledge of prices, load and solar.

    Variables per step t: c_t, d_t >= 0 (charge/discharge, grid side, kW), u_t in {0,1} (1 = charging allowed),
    imp_t, exp_t >= 0 (grid import/export, kW), s_t (charge after step t, kWh).
    Same limits as the environment: |p| <= p_max, 0 <= s <= capacity, s_0 = s_96 = soc0, no simultaneous
    charge and discharge. Optional cap on equivalent full cycles for the constrained comparison.
    """
    b, tar = env_cfg.battery, env_cfg.tariff
    T, dt = STEPS_PER_DAY, b.dt_h
    price = np.asarray(price, float)
    base = np.asarray(base_kw, float)
    nv = 6 * T  # c, d, u, imp, exp, s
    C, D, U, I, E, S = (np.arange(T) + k * T for k in range(6))

    obj = np.zeros(nv)
    obj[I] = (price + tar.import_adder) * dt
    obj[E] = -(price + tar.export_adder) * dt
    obj[C] += env_cfg.wear_cost * dt
    obj[D] += env_cfg.wear_cost * dt

    rows, lb, ub = [], [], []
    A = lil_matrix((4 * T + 1, nv))
    r = 0
    for t in range(T):
        # grid balance: imp - exp - c + d = base
        A[r, I[t]] = 1; A[r, E[t]] = -1; A[r, C[t]] = -1; A[r, D[t]] = 1
        lb.append(base[t]); ub.append(base[t]); r += 1
        # storage: s_t - s_{t-1} - eta*dt*c + dt/eta*d = 0   (s_{-1} = soc0)
        A[r, S[t]] = 1
        if t > 0:
            A[r, S[t - 1]] = -1
        A[r, C[t]] = -b.eta * dt; A[r, D[t]] = dt / b.eta
        rhs = b.soc0 if t == 0 else 0.0
        lb.append(rhs); ub.append(rhs); r += 1
        # c <= P u ; d <= P (1 - u)
        A[r, C[t]] = 1; A[r, U[t]] = -b.p_max_kw
        lb.append(-np.inf); ub.append(0.0); r += 1
        A[r, D[t]] = 1; A[r, U[t]] = b.p_max_kw
        lb.append(-np.inf); ub.append(b.p_max_kw); r += 1
    # throughput cap (equivalent full cycles)
    if efc_cap is not None:
        A[r, C] = dt; A[r, D] = dt
        lb.append(-np.inf); ub.append(efc_cap * 2 * b.capacity_kwh)
    else:
        lb.append(-np.inf); ub.append(np.inf)

    lo = np.zeros(nv); hi = np.full(nv, np.inf)
    hi[C] = b.p_max_kw; hi[D] = b.p_max_kw; hi[U] = 1
    hi[S] = b.capacity_kwh
    lo[S[-1]] = hi[S[-1]] = b.soc0  # end of day back at soc0
    integrality = np.zeros(nv); integrality[U] = 1

    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lb, ub), bounds=Bounds(lo, hi),
               integrality=integrality, options={"mip_rel_gap": 1e-9, "time_limit": 60})
    if not res.success:
        raise RuntimeError(f"oracle MILP failed: {res.message}")
    x = res.x
    p = x[C] - x[D]
    bill = float(np.sum(tar.cost(price, (base + p) * dt)))
    return OracleResult(bill=bill, p_kw=p, efc=float(np.sum(np.abs(p)) * dt / (2 * b.capacity_kwh)))


def day_arrays(env: HomeBatteryEnv, day: int):
    env = env.unwrapped
    i0 = env.day_start[day]
    sl = slice(i0, i0 + STEPS_PER_DAY)
    base = (env.load[sl] - env.solar[sl]) if env.cfg.include_load_solar else np.zeros(STEPS_PER_DAY)
    return env.price[sl], base


def run_controller(env: HomeBatteryEnv, day: int, controller) -> dict:
    """Roll out one day. controller(env, obs) returns either ('kw', value) or ('action', array)."""
    obs, _ = env.reset(options={"day": day})
    done = False
    while not done:
        kind, val = controller(env, obs)
        if kind == "kw":
            obs, _, done, _, info = env.unwrapped.step_kw(val)
        else:
            obs, _, done, _, info = env.step(val)
    return info


def threshold_controller(p_low: float, p_high: float):
    """Charge at full power below p_low, discharge at full power above p_high ($/kWh), otherwise idle.

    The environment's end-of-day rule clips the request, exactly as it does for the RL agents.
    """
    def ctrl(env: HomeBatteryEnv, obs):
        env = env.unwrapped
        price = env.price[env._idx()]
        pm = env.cfg.battery.p_max_kw
        return "kw", (pm if price <= p_low else (-pm if price >= p_high else 0.0))

    return ctrl


def tune_threshold(env: HomeBatteryEnv, days, quantiles=np.linspace(0.05, 0.95, 19)) -> tuple[float, float, float]:
    """Grid search over price quantiles of the given (training) days. Returns (p_low, p_high, mean bill)."""
    env = env.unwrapped
    prices = np.concatenate([day_arrays(env, d)[0] for d in days])
    qs = np.quantile(prices, quantiles)
    best = (None, None, np.inf)
    for i, lo in enumerate(qs):
        for hi in qs[i + 1:]:
            ctrl = threshold_controller(lo, hi)
            mean_bill = np.mean([run_controller(env, d, ctrl)["bill"] for d in days])
            if mean_bill < best[2]:
                best = (float(lo), float(hi), float(mean_bill))
    return best
