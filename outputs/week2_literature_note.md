# Week 2 — Literature Sanity Check

**Question this note answers:** is a 75.6% drop in average waiting time on sumo-rl's
2-way single-intersection benchmark a believable result, or a sign of a broken
experiment?

**Short answer:** believable, and the benchmark's own demand file explains why the
margin is this large.

## The result

3 independently trained PPO seeds, evaluated deterministically
(`model.predict(..., deterministic=True)`), 3600 simulated seconds each.

| Metric | Fixed-time | PPO (3-seed mean) | Change |
|---|---|---|---|
| Avg wait time (s) | 16.4 | 4.0 | **−75.6%** |
| Peak queue (vehicles) | 65.7 | 22.0 | **−66.5%** |
| Throughput (vehicles) | 2089 | 2430 | **+16.3%** |

Source: `outputs/week2_benchmark_metrics.csv`, chart in
`outputs/week2_benchmark_comparison.png`.

## Why the margin is large — read the demand file

`single-intersection-vhvh.rou.xml` ("vertical-horizontal-vertical-horizontal")
defines flows in 25000-second blocks. In the first block, which is the only one a
3600 s episode ever reaches:

- north-south movements: 350 + 350 + 300 + 300 + 350 + 350 = **2000 veh/h**
- east-west movements: 100 + 100 + 50 + 50 + 100 + 100 = **500 veh/h**

So demand is roughly **4:1 in favour of the north-south axis**.

The static `tlLogic` shipped in `single-intersection.net.xml` splits green time
**symmetrically** between the two axes — 33 s of through-green each, plus 6 s turn
phases and 2 s yellows, for an 86 s cycle. A fixed program giving equal green to a 4:1
demand split is badly matched to this demand by construction, so a controller that can
reallocate green time has a lot of headroom to recover. A large improvement is the
expected outcome here, not an anomaly.

The throughput figure supports this reading rather than contradicting it. Total demand
over the episode is about 2500 vehicles. Fixed-time clears 2089 (~84%) and leaves the
rest queued; PPO clears 2430 (~97%), which is close to the ceiling. Throughput improves
by "only" 16% because it *cannot* improve much more — the remaining metric with room to
move is delay, and that is where the 76% appears.

## Comparison with published behaviour

Qualitative agreement, on three points that can be checked directly against the
installed `sumo-rl` package and its repository:

1. **Direction and rough magnitude.** sumo-rl ships this network specifically as a
   learning benchmark and its own example experiments (DQN, A3C, tabular Q-learning on
   `2way-single-intersection`) are built around the same `diff-waiting-time` reward
   used here. Large delay reductions against the static program are the reported
   behaviour for this benchmark, not an outlier.
2. **Convergence speed.** A single intersection with a 21-dimensional observation and
   4 discrete actions is a small control problem. The reward curve
   (`outputs/monitor/seed{N}.monitor.csv`) flattens well before the 100k-step budget,
   which is consistent with this benchmark being used as a fast smoke-test task in the
   traffic-RL literature rather than a hard one.
3. **Seed spread.** The three seeds land at 4.35 / 4.29 / 3.36 s average wait — same
   ordering, same magnitude, no seed collapsed. Reported results for this benchmark are
   similarly tight; wide seed variance here would have indicated a wiring problem.

**Pending verification.** No specific published figure is quoted in this note. Exact
numbers and DOIs from the primary sources (LucasAlegre's sumo-rl experiments and the
RESCO benchmark suite) are verified against the papers themselves in Week 12, per the
project plan's citation-verification step. Claiming a numeric match before doing that
check would be the kind of unverified citation that step exists to prevent.

## Caveats worth stating in a viva

- **Only the first demand block is exercised.** The `vhvh` file's whole point is that
  the heavy axis *switches* at t=25000 s. A 3600 s episode never sees the switch, so
  this week tested adaptation to a *fixed* asymmetric demand, not adaptation to a
  changing one. Robustness to demand shift is tested properly in Week 6 with the
  `asymmetric` scenario and in Week 5's online-learning loop.
- **The baseline is the network's stock program, not a tuned one.** A traffic engineer
  retiming that intersection for a 4:1 split would close much of this gap. The honest
  claim is "PPO beats the shipped fixed-time program", not "PPO beats good signal
  engineering".
- **Single intersection.** Nothing here says anything about coordination; that is
  Weeks 4-6.
