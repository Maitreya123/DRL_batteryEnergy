## Results: retail tariff, SAC and day-ahead forecasts

*Draft by Srikar. All runs below use the main configuration (3M steps, battery-only reward, linear actions,
initial log std −1, γ = 1) and were trained on a MacBook (Python 3.13, Stable-Baselines3 2.9.2). Scores are the
share of the hindsight oracle's savings on the 91 held-out test days; ± is a 95% t-interval over seeds.*

### A charge on imports makes load and solar matter

Under the symmetric tariff, the household's own load and solar cannot change the optimal battery schedule, which
was the main caveat of our core results. We added a 4.5 ¢/kWh charge on imports (a delivery charge) and
retrained PPO and PPO-Lagrangian (budget 0.75), 5 seeds each.

| Method | % of oracle, all days | ordinary days | cycles/day |
|---|---|---|---|
| PPO | 45.6 ± 12.5 | 47.6 ± 9.3 | 0.73 |
| PPO-Lagrangian, budget 0.75 | 39.3 ± 9.5 | 44.0 ± 10.6 | 0.68 |
| Threshold rule | 32.0 | 11.2 | 0.59 |
| Oracle | 100 | 100 | 0.89 |

The import charge makes cycling much less worthwhile: the oracle cycles half as much (0.89 instead of 1.72 times a
day) and saves $0.84 instead of $0.93 a day. Here PPO clearly beats the price-only threshold rule on ordinary days (47.6% vs 11.2%),
where under the symmetric tariff the two were tied. The rule ignores solar and buys from the grid regardless; PPO
evidently uses the load and solar in its state. The agents are, however, further from the oracle than under the
symmetric tariff, and vary more across seeds. The 0.75 budget barely binds, since PPO alone already cycles 0.73
times a day, so the constraint mostly costs savings here.

### A second continuous-control algorithm: SAC

| Method (5 seeds) | % of oracle, all days | ordinary days | cycles/day |
|---|---|---|---|
| PPO | 71.8 ± 0.9 | 57.9 ± 2.5 | 0.95 |
| DQN, 11 discrete levels | 64.5 ± 7.5 | 44.6 ± 12.7 | 1.54 |
| SAC | 52.1 ± 15.4 | 38.3 ± 6.7 | 1.52 |
| Threshold rule | 56.2 | 58.2 | 0.59 |

SAC, with Stable-Baselines3's defaults and no tuning, falls short of PPO and of the threshold rule. Like DQN, it
cycles about 1.5 times a day, close to the oracle's 1.72, without the oracle's savings to show for it. Both
off-policy, bootstrapped learners do worse than PPO here. One plausible reason is γ = 1, which suits PPO's
Monte-Carlo-like advantage estimates over a 96-step day better than a bootstrapped critic. We did not test this.

### Day-ahead price forecasts

ERCOT publishes day-ahead hub prices the afternoon before each delivery day. We added the day-ahead price of each
of the next 24 hours to the state, set to 0 for hours past midnight and for prices not yet published, and
retrained PPO alongside a fresh PPO baseline on the same machine.

| Method (5 seeds) | % of oracle, all days | ordinary days | cycles/day |
|---|---|---|---|
| PPO | 70.9 ± 2.9 | 56.2 ± 4.6 | 1.00 |
| PPO + 24 h of day-ahead prices | 70.3 ± 5.9 | 57.9 ± 5.4 | 1.01 |

The forecasts do not measurably help, and PPO still only ties the threshold rule on ordinary days. On ordinary
days the day-ahead price predicts the typical shape of the day (hourly averages correlate at 0.90) but individual
real-time prices poorly (0.18 per interval, against 0.39 for the real-time price an hour earlier). The shape is
something the agent already learns from the time of day.

### Why PPO only ties the rule on ordinary days

Hour by hour, on the 87 ordinary test days, PPO earns $0.23 a day more than the rule between 14:00 and 20:00 by
discharging into moderate peaks, then loses $0.20 a day between 22:00 and midnight refilling the battery to 50%.
It is idle in only 20% of steps (the rule: 86%), sells at 4.3 ¢/kWh on average against the rule's 5.9 ¢, and earns
$0.29 per cycle against the rule's $0.51. PPO's shortfall is a thin margin per cycle and an expensive refill
before the end of the day, not missed peaks.

### Smaller experiments

- **Ablation** (3 seeds each): only the battery-only reward changes the mean noticeably (about 3 points). Removing
  all three fixes made one of three seeds collapse to 26% of the oracle, so the fixes mainly buy reliability.
- **The 0.5 budget**: neither a longer λ warm-up nor a PID multiplier update reaches it. Small flows under 1 kW
  account for 0.09-0.18 cycles a day, because idle is not a fixed action under linear actions. A 1 kW deadband at
  evaluation time brings the agents to 0.49-0.54 cycles a day for a small loss in savings.
- **Wear cost**: seeds 3 and 4 were added for every wear cost, so each now has 5 seeds.
