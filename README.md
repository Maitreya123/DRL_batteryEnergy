# Constrained deep RL for home battery control under ERCOT prices

CSCE-642 (Deep Reinforcement Learning) course project, Texas A&M, Fall 2026.
Maitreya Cherukupalli and Srikar Sankranti.

An agent controls a 13.5 kWh / 5 kW home battery with 5 kW of rooftop solar in Austin, Texas, buying and selling
at ERCOT real-time prices in 15-minute steps. We compare PPO, PPO-Lagrangian with a battery-cycling budget, and PPO
with a fixed wear cost against a tuned price-threshold rule and a perfect-foresight MILP oracle.

## Results (91 held-out test days, 95% CI over seeds)

| Method | % of oracle savings, all days | ordinary days | cycles/day |
|---|---|---|---|
| PPO (5 seeds) | 71.8 ± 0.9 | 57.9 ± 2.5 | 0.95 |
| PPO-Lagrangian, budget 0.75 cycles/day (5 seeds) | 70.7 ± 1.8 | 55.7 ± 4.5 | 0.74 |
| PPO + wear cost 1 ¢/kWh (3 seeds) | 67.8 ± 15.4 | 53.3 ± 19.4 | 0.70 |
| DQN, 11 discrete actions (5 seeds) | 64.5 ± 7.5 | 44.6 ± 12.7 | 1.54 |
| SAC (5 seeds) | 52.1 ± 15.4 | 38.3 ± 6.7 | 1.52 |
| Threshold rule | 56.2 | 58.2 | 0.59 |
| Oracle (hindsight) | 100 | 100 | 1.72 |

DQN uses the same configuration as PPO, with the action range split into 11 evenly spaced levels
(`results/dqn_test_table.txt`). It cycles far more than PPO and trails it on ordinary days.

SAC uses the same environment and flags as PPO (`--algo sac`, Stable-Baselines3 defaults otherwise, no tuning;
`results/sac_test_table.txt`). It falls short of both PPO and the threshold rule, cycling about 1.5 times a day
without the oracle's savings to show for it; with gamma = 1.0 its bootstrapped critic may be the weak point.

"Ordinary days" excludes the 4 test days whose price exceeded $1/kWh. Full tables: `results/test_table.txt`,
`results/spikes_table.txt`. Figures: `results/figures/`. Day replays: `results/videos/`.

## Retail tariff (4.5 ¢/kWh charge on imports)

Under the symmetric tariff, the household's own load and solar cannot change the best battery schedule. With a
charge on imports, storing solar instead of exporting it starts to pay, so this checks whether the agents use the
load and solar in their state. Same configuration as above plus `--import-adder 0.045`; 91 test days, 95% CI over
seeds. Full table: `results/retail_test_table.txt`.

| Method | % of oracle savings, all days | ordinary days | cycles/day |
|---|---|---|---|
| PPO (5 seeds) | 45.6 ± 12.5 | 47.6 ± 9.3 | 0.73 |
| PPO-Lagrangian, budget 0.75 (5 seeds) | 39.3 ± 9.5 | 44.0 ± 10.6 | 0.68 |
| Threshold rule | 32.0 | 11.2 | 0.59 |
| Oracle (hindsight) | 100 | 100 | 0.89 |

The import charge makes cycling less profitable: the oracle drops from 1.72 to 0.89 cycles/day. PPO now clearly beats
the price-only threshold rule on ordinary days (47.6% vs 11.2%), where under the symmetric tariff the two were
tied, but it is further from the oracle than before, and the seeds vary more. A 0.75 cycle budget barely binds here,
since unconstrained PPO already cycles 0.73 times a day, so the constraint mostly costs savings. The threshold rule
is tuned on price alone and charges from the grid regardless of solar, so it is a weak baseline under this tariff.

## Day-ahead price forecasts in the state

ERCOT publishes day-ahead (DAM) hub prices for each delivery day the afternoon before. With `--dam-hours 24`, the
state also holds the DAM price for each of the next 24 hours: 0 for hours past midnight and for prices not yet
published. Both groups below were trained on a MacBook with the main configuration, 5 seeds each
(`results/dam_test_table.txt`).

| Method | % of oracle savings, all days | ordinary days | cycles/day |
|---|---|---|---|
| PPO | 70.9 ± 2.9 | 56.2 ± 4.6 | 1.00 |
| PPO + 24 h of DAM prices | 70.3 ± 5.9 | 57.9 ± 5.4 | 1.01 |
| Threshold rule | 56.2 | 58.2 | 0.59 |

The forecasts do not measurably help: the difference is well inside the seed-to-seed spread, and PPO still only
ties the threshold rule on ordinary days. A likely reason: on ordinary days the DAM price predicts the typical shape
of the day well (hourly averages correlate at 0.90) but individual real-time prices poorly (0.18 per interval, against
0.39 for the real-time price an hour earlier). The shape is something the agent already learns from the time of day.
The DAM file is downloaded by `scripts/download_data.py`.

## Layout

```
src/battery_rl/
  data.py        ERCOT + ResStock + pvlib solar merged on a 15-min UTC grid; week-based train/val/test split
  env.py         Gymnasium environment (one day = 96 steps, start and end at 50% charge)
  baselines.py   MILP oracle (scipy/HiGHS), threshold rule
  evaluate.py    per-day scoring against no battery and the oracle (oracle results cached)
  train.py       DQN, PPO, PPO-Lagrangian, SAC (Stable-Baselines3)
scripts/
  download_data.py     fetch the raw ERCOT, weather and load data and build the dataset
  check_baselines.py   env-vs-oracle replay check and baseline savings
  report.py            test-day table with seed-level confidence intervals
  spikes.py            stretch goal: price-spike robustness
  plots.py             report figures
  animate.py           day-replay videos
  schedules.py         hour-by-hour comparison of PPO, the threshold rule and the oracle
```

## Reproducing

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Training is faster on CPU than GPU for these small
networks (measured 6-7x), so runs are parallelised across cores instead.

```bash
uv sync --python 3.11 --managed-python

# raw data (not in the repo): ERCOT 2018 real-time and day-ahead hub prices, CityLearn's Austin AMY2018 weather
# file and the ResStock household load, all public. Downloads into data/raw/ and builds data/processed/.
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/check_baselines.py

# main configuration (one seed shown)
C="--steps 3000000 --n-envs 4 --lr 3e-4 --net 64 --eval-every 250000 \
   --battery-reward --linear-actions --log-std-init -1 --gamma 1.0"
.venv/bin/python -m battery_rl.train --algo ppo $C --seed 0 --name ppo_s0
.venv/bin/python -m battery_rl.train --algo ppo-lag --efc-budget 0.75 $C --seed 0 --name ppolag_b0.75_s0

.venv/bin/python scripts/report.py --group PPO ppo_s0 --group PPO-Lag ppolag_b0.75_s0
```

The other configurations add one flag to `$C`: `--efc-budget 1.0` / `0.5` for the other budgets, `--wear-cost W`
(with `--algo ppo`) for the wear-cost runs, and `--exclude-spikes 1.0` to train without price-spike days.
`scripts/plots.py` expects the run names it lists in `GROUPS` (`ppo_s{0-4}`, `ppolag_b{budget}_s{0-4}`,
`ppowear_w{cost}_s{0-2}`); `scripts/animate.py` also needs `ffmpeg` on the path.

```bash
.venv/bin/python scripts/spikes.py --group PPO ppo_s0 ppo_s1 --group "PPO (no spikes)" noSpike_ppo_s0 noSpike_ppo_s1
.venv/bin/python scripts/plots.py
.venv/bin/python scripts/animate.py --day 332 --run ppo_s4
```

## Tests

```bash
uv sync --python 3.11 --managed-python --group dev
.venv/bin/python -m pytest tests
```

The tests use four synthetic days, so they need none of the raw data. They check that the environment and the MILP
oracle agree exactly on bill, cycles and final charge, that any action sequence keeps the battery within its limits
and back at 50% by midnight, and that the Lagrangian penalty reaches the training environments.

## Why PPO only ties the threshold rule on ordinary days

`scripts/schedules.py` replays PPO (5 seeds, trained on a MacBook with the main configuration), the tuned threshold
rule and the oracle on the 87 ordinary test days and compares them hour by hour (`results/schedules_summary.txt`,
`results/schedules_by_hour.csv`, `results/figures/schedules.png`). PPO and the rule end up with almost the same
savings ($0.283 vs $0.293 a day, 56% vs 58% of the oracle), but they get there differently:

- **PPO earns more in the afternoon and evening** (14:00-20:00, +$0.23/day over the rule): it discharges
  into moderate peaks that the rule's fixed 5.4 ¢ threshold ignores.
- **It gives it back at the end of the day** (22:00-24:00, -$0.20/day): having discharged in the evening, it has
  to recharge to 50% before midnight, when prices are still above the overnight lows.
- **It trades small amounts almost all the time.** PPO is idle in 20% of steps and at full power in 3%, against
  86% and 12% for the rule. It sells at 4.3 ¢/kWh on average (the rule: 5.9 ¢) and earns $0.29 per cycle against
  the rule's $0.51. That is about the oracle's $0.29 per cycle, but at 0.98 cycles a day instead of 1.72.

So PPO's problem on ordinary days is not missing the peaks. It cycles at a thin margin, and most of what it
loses against the rule comes from refilling the battery late in the evening to meet the end-of-day rule.

## Changes from the proposal

- **Actions** are a fraction of the power feasible at that step (mapped linearly onto the feasible range), not a
  fixed ±5 kW. With fixed actions, an empty or full battery makes most of the action range do nothing, and PPO got
  stuck there.
- **Reward** counts only the battery's effect on the bill. Under the symmetric tariff the household's own cost is
  unaffected by any action, and including it made the reward too noisy for PPO to learn reliably.
- **PPO-Lagrangian** holds λ at 0 for the first 500k steps, and only selects checkpoints from after that.
- **Solar** comes from CityLearn's hourly AMY2018 Austin weather file through pvlib, not NSRDB.
- **Budgets** of 0.5 and 0.75 cycles/day were added, because unconstrained PPO already cycles about once a day.

## Limitations

- **Price information.** The agent, like the threshold rule, sees the real-time price of the interval it is
  acting in. In practice that price is only settled after the interval, so the reported savings are an upper
  bound for a controller that acts on forecasts.
- **One household, one year.** All results are for a single Travis County home and 2018 prices; the test days
  are held-out weeks of the same year, not a later year.
- **Checkpoint selection** uses the share of oracle savings summed over validation days, which a few spike days
  can dominate.

## AI-use disclosure

As the course policy allows for the project, we used Claude (Anthropic) to help write the code and run the
experiments. We reviewed the code and results, and every number above comes from runs in this repository.
