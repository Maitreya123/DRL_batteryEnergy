## Related work

*Draft by Srikar, adapted from our literature survey. Reference numbers follow the survey's list.*

Our project draws on three bodies of work: optimization-based valuation of energy storage, reinforcement learning
for storage and building energy control, and constrained reinforcement learning.

**Optimization-based arbitrage and battery aging.** Sioshansi et al. [10] value a price-taking storage device in
PJM by optimizing its schedule against historical prices. Our oracle is the same kind of hindsight optimization,
solved per day for one household, with the household's load and solar included. Like [10], we treat the battery as
a price taker. Where [10] measures how much money is available, we measure how much of it a policy that cannot see
the future actually captures. Krishnamurthy et al. [11] instead solve a scenario-based stochastic program over
day-ahead and real-time prices. That is the model-based alternative to our model-free agent, which acts only in
the real-time market, as a household on a wholesale-indexed plan would. Our day-ahead experiment borrows its
information structure: day-ahead prices are published the afternoon before delivery and can inform real-time
decisions. Xu et al. [12] price battery aging by cycle depth. We count throughput as equivalent full cycles
instead, which ignores depth; we list this as a limitation.

**Reinforcement learning for storage and home energy.** Cao et al. [13] learn grid-scale arbitrage on UK prices
with a noisy-network DQN and a detailed degradation model in the reward. Our fixed-wear-cost PPO runs follow the
same "degradation as a penalty" design, while our core experiment treats wear as a constraint with a budget,
which avoids hand-pricing degradation. Xu et al. [14] use a recurrent network to summarize PJM price history,
because the current price alone is not Markov. We take a simpler route: the last hour of prices and time features
in the state, plus, in one experiment, the published day-ahead prices for the coming hours. Kell et al. [15] are
closest to our setting: a home battery with rooftop solar, controlled by DDPG with continuous power. We differ in
using PPO and PPO-Lagrangian as the main agents, in reporting every result as a share of a hindsight-optimal
oracle on the same model, and in constraining battery cycling. We also report an off-policy actor-critic
comparison of our own, SAC [3]. Vázquez-Canteli and Nagy [16] review RL for demand response; our study is the
single-household case of that setting. Their CityLearn environment [17] served as a reference design, and its
Austin AMY2018 weather file is our solar input.

**Deep RL algorithms.** We use Stable-Baselines3 [4] for every agent, so differences come from the formulation,
not the implementation. PPO [2] is our main agent: an on-policy method that limits each policy update with a
clipped probability-ratio objective. DQN [1] serves as a value-based comparison on 11 discretized power levels.
SAC [3], an off-policy maximum-entropy actor-critic, is usually more sample-efficient than PPO on continuous
control. Our simulator is cheap, so sample efficiency matters less here than final performance.

**Constrained RL.** Our core experiment is a constrained MDP [5]: maximize expected savings subject to a bound on
expected equivalent full cycles per day. CPO [6] solves CMDPs with a trust-region update and a linearized cost
constraint. We use the simpler Lagrangian approach of RCPO [7] and Ray et al. [8]: the cycle cost, weighted by a
multiplier, is subtracted from the reward, and the multiplier rises while the agent overspends its budget. Ray et
al. found this PPO-Lagrangian combination a strong, simple baseline on Safety Gym. Stooke et al. [9] show that the
plain multiplier update acts like integral control and oscillates, and add proportional and derivative terms. We
test their PID update on our tightest budget, where the plain update misses the constraint.

**Data.** Prices are ERCOT's 2018 real-time hub average [21] and, for the forecast experiment, ERCOT's day-ahead
hub prices. Household load is a Travis County single-family home from the End-Use Load Profiles dataset [20],
simulated with actual 2018 weather. Rooftop solar output comes from pvlib [19], driven by the AMY2018 Camp Mabry
weather file that CityLearn [17] distributes. The proposal planned NSRDB [18] instead. Both are actual 2018
weather, so solar still lines up day by day with prices and load.
