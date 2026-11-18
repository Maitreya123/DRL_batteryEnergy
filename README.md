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

## Layout

```
src/battery_rl/
  data.py        ERCOT + ResStock + pvlib solar merged on a 15-min UTC grid; week-based train/val/test split
  env.py         Gymnasium environment (one day = 96 steps, start and end at 50% charge)
  baselines.py   MILP oracle (scipy/HiGHS), threshold rule
  evaluate.py    per-day scoring against no battery and the oracle (oracle results cached)
  train.py       DQN, PPO, PPO-Lagrangian, SAC (Stable-Baselines3)
scripts/
  check_baselines.py   env-vs-oracle replay check and baseline savings
  report.py            test-day table with seed-level confidence intervals
  spikes.py            stretch goal: price-spike robustness
  plots.py             report figures
  animate.py           day-replay videos
```

## Reproducing

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Training is faster on CPU than GPU for these small
networks (measured 6-7x), so runs are parallelised across cores instead.

```bash
uv sync --python 3.11 --managed-python

# raw data (not in the repo): ERCOT 2018 RTM prices and CityLearn's Austin AMY2018 weather file
#   data/raw/ercot/   <- RTMLZHBSPP_2018.zip from ERCOT report NP6-785-ER, unzipped
#   data/raw/citylearn/weather.epw  <- CityLearn data/datasets/tx_travis_county_neighborhood/weather.epw
# ResStock load is downloaded automatically from the OEDI data lake.
.venv/bin/python -m battery_rl.data
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

## Changes from the proposal

- **Actions** are a fraction of the power feasible at that step (mapped linearly onto the feasible range), not a
  fixed ±5 kW. With fixed actions, an empty or full battery makes most of the action range do nothing, and PPO got
  stuck there.
- **Reward** counts only the battery's effect on the bill. Under the symmetric tariff the household's own cost is
  unaffected by any action, and including it made the reward too noisy for PPO to learn reliably.
- **PPO-Lagrangian** holds λ at 0 for the first 500k steps, and only selects checkpoints from after that.
- **Solar** comes from CityLearn's hourly AMY2018 Austin weather file through pvlib, not NSRDB.
- **Budgets** of 0.5 and 0.75 cycles/day were added, because unconstrained PPO already cycles about once a day.

## Ablation: what each fix from "Changes from the proposal" buys

Each row removes one fix from the main configuration (PPO rows) or from PPO-Lagrangian with a 0.75 budget (last
two rows), 3 seeds each, trained on a MacBook; "original setup" removes all three PPO fixes at once (absolute
actions, full-bill reward, initial std 1). Full table: `results/ablation_table.txt`.

| Configuration | % of oracle, all days | ordinary days | cycles/day | per-seed, all days |
|---|---|---|---|---|
| PPO, all fixes | 69.5 ± 4.8 | 53.6 ± 3.6 | 1.01 | |
| without battery-only reward | 66.9 ± 3.4 | 49.7 ± 2.1 | 0.94 | 67.0, 68.2, 65.5 |
| without linear actions | 71.0 ± 5.8 | 56.1 ± 10.3 | 0.95 | 73.6, 69.0, 70.5 |
| without the calmer initial policy | 72.4 ± 1.4 | 58.7 ± 5.1 | 1.02 | 72.2, 73.0, 71.9 |
| original setup (none of the three) | 56.0 ± 64.2 | 42.8 ± 51.3 | 1.02 | 72.4, **26.2**, 69.5 |
| PPO-Lagrangian b0.75, with λ warm-up | 71.1 ± 4.3 | 54.4 ± 2.3 | 0.72 | |
| without λ warm-up | 70.5 ± 5.2 | 55.8 ± 9.8 | 0.70 | |

With these 3M-step runs, no single fix changes the mean much except the battery-only reward, which is worth about
3 points (4 on ordinary days) consistently across seeds. Removing all three at once is what breaks training: one of
three seeds collapsed to 26% of the oracle while the other two were normal, which matches the "PPO sometimes gets
stuck" failure the fixes were made for. So the fixes mainly buy reliability across seeds, not a higher mean. The λ
warm-up has no measurable effect at a 0.75 budget. Three seeds cannot rule out small effects either way.

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
