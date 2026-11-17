"""Checks on synthetic data, so they run without the ERCOT / ResStock / weather files.

    .venv/bin/python -m pytest tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rl.baselines import day_arrays, run_controller, solve_oracle, threshold_controller
from battery_rl.env import STEPS_PER_DAY, DiscreteActions, EnvConfig, HomeBatteryEnv, Tariff
from battery_rl.evaluate import summarize

N_DAYS = 4


def synthetic_data(seed: int = 0) -> pd.DataFrame:
    """N_DAYS days with an evening price peak, one negative-price dip, one spike, household load and solar."""
    rng = np.random.default_rng(seed)
    n = N_DAYS * STEPS_PER_DAY
    step = np.tile(np.arange(STEPS_PER_DAY), N_DAYS)
    hour = step / 4
    price = 0.025 + 0.03 * np.exp(-((hour - 18) ** 2) / 4) + 0.005 * rng.standard_normal(n)
    price[STEPS_PER_DAY + 10: STEPS_PER_DAY + 14] = -0.02  # negative prices
    price[2 * STEPS_PER_DAY + 70] = 5.0  # scarcity spike
    load = 1.0 + 0.5 * rng.random(n)
    solar = np.clip(4.0 * np.sin(np.pi * (hour - 7) / 12), 0.0, None)
    idx = pd.date_range("2018-01-01 06:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"price": price, "load": load, "solar": solar,
                         "day": np.repeat(np.arange(N_DAYS), STEPS_PER_DAY), "step": step,
                         "dow": np.repeat(np.arange(N_DAYS) % 7, STEPS_PER_DAY)}, index=idx)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic_data()


@pytest.mark.parametrize("tariff", [Tariff(), Tariff(import_adder=0.045)])
@pytest.mark.parametrize("day", range(N_DAYS))
def test_oracle_replay_matches_env(data, tariff, day):
    """The MILP and the environment must model the same battery, tariff and end-of-day rule."""
    cfg = EnvConfig(tariff=tariff)
    env = HomeBatteryEnv(data, np.arange(N_DAYS), cfg)
    price, base = day_arrays(env, day)
    orc = solve_oracle(price, base, cfg)
    sched = iter(orc.p_kw)
    info = run_controller(env, day, lambda e, o: ("kw", next(sched)))
    assert info["bill"] == pytest.approx(orc.bill, abs=1e-6)
    assert info["final_soc"] == pytest.approx(cfg.battery.soc0, abs=1e-6)
    assert info["efc"] == pytest.approx(orc.efc, abs=1e-6)
    assert orc.bill <= info["bill_no_battery"] + 1e-9  # idling is always feasible


def test_oracle_cycle_cap(data):
    cfg = EnvConfig()
    price, base = day_arrays(HomeBatteryEnv(data, np.arange(N_DAYS), cfg), 0)
    free = solve_oracle(price, base, cfg)
    capped = solve_oracle(price, base, cfg, efc_cap=0.25)
    assert capped.efc <= 0.25 + 1e-6
    assert capped.bill >= free.bill - 1e-9


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_random_policy_respects_limits(data, linear, seed):
    """Any action sequence keeps 0 <= SoC <= capacity, |p| <= p_max, and ends the day at soc0."""
    cfg = EnvConfig(linear_actions=linear)
    env = HomeBatteryEnv(data, np.arange(N_DAYS), cfg)
    env.action_space.seed(seed)
    b = cfg.battery
    obs, _ = env.reset(seed=seed)
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        obs, _, done, _, info = env.step(env.action_space.sample() * 3)  # also out-of-range raw actions
        assert -1e-9 <= info["soc"] <= b.capacity_kwh + 1e-9
        assert abs(info["p_kw"]) <= b.p_max_kw + 1e-9
    assert info["final_soc"] == pytest.approx(b.soc0, abs=1e-6)


def test_action_mapping_endpoints(data):
    env = HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig())
    env.reset(options={"day": 0})
    lo, hi = env.cfg.battery.feasible_power(env.soc, STEPS_PER_DAY)
    assert env.action_to_kw(-1.0) == pytest.approx(lo)
    assert env.action_to_kw(1.0) == pytest.approx(hi)
    assert env.action_to_kw(0.0) == pytest.approx(0.0)  # idle allowed at the start of the day


def test_battery_only_reward_differs_by_household_cost(data):
    """battery_only_reward removes a term no action can change, so the two rewards differ by the no-battery cost."""
    totals = {}
    for flag in (False, True):
        env = HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig(battery_only_reward=flag))
        env.reset(options={"day": 1})
        total, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(np.array([0.3]))
            total += r
        totals[flag] = (total, info)
    (r_full, info), (r_batt, _) = totals[False], totals[True]
    assert r_full - r_batt == pytest.approx(-env.cfg.reward_scale * info["bill_no_battery"], abs=1e-9)


def test_threshold_rule_idles_between_thresholds(data):
    env = HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig())
    info = run_controller(env, 0, threshold_controller(-np.inf, np.inf))
    assert info["bill"] == pytest.approx(info["bill_no_battery"])
    assert info["efc"] == 0.0


def test_discrete_actions_cover_feasible_range(data):
    env = DiscreteActions(HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig()))
    assert env.action_space.n == 11
    assert env.action(0)[0] == -1.0 and env.action(10)[0] == 1.0 and env.action(5)[0] == 0.0


def test_summarize_ignores_days_without_oracle_saving():
    per_day = pd.DataFrame({"bill": [1.0, 2.0, 3.0], "no_battery": [2.0, 3.0, 3.0], "oracle": [0.0, 1.0, 3.0],
                            "efc": [1.0, 1.0, 0.0]})
    s = summarize(per_day)
    assert s["pct_of_oracle"] == pytest.approx(50.0)
    assert s["median_day_pct_of_oracle"] == pytest.approx(50.0)  # the third day has no saving to share


def test_lagrangian_penalty_reaches_monitored_envs(data):
    """set_lambda must pass through the Monitor wrapper that make_vec_env adds, and charge lambda per cycle."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    from battery_rl.train import LagrangianPenalty

    venv = make_vec_env(lambda: LagrangianPenalty(HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig())),
                        n_envs=2, seed=0, vec_env_cls=DummyVecEnv)
    venv.env_method("set_lambda", 2.0)
    assert venv.get_attr("lam") == [2.0, 2.0]

    env = LagrangianPenalty(HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig()))
    plain = HomeBatteryEnv(data, np.arange(N_DAYS), EnvConfig())
    env.set_lambda(2.0)
    env.reset(options={"day": 0}); plain.reset(options={"day": 0})
    _, r_pen, _, _, info = env.step(np.array([1.0]))
    _, r_plain, _, _, _ = plain.step(np.array([1.0]))
    assert r_plain - r_pen == pytest.approx(2.0 * info["throughput_kwh"] / (2 * 13.5))
