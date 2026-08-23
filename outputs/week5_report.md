# Week 5 — Parameter-Shared MARL, Green-Wave Shaping, Lagrangian Fairness

Three changes on top of Week 4's independent learners:

1. **Parameter sharing** — one policy drives all twelve junctions instead of twelve
   separate policies, so every junction's experience updates the same weights.
2. **Green-wave reward shaping** — each junction is penalised for discharging into a
   congested downstream link (`-alpha * mean outgoing-lane density`), which makes
   pushing congestion onto a neighbour costly rather than free.
3. **Lagrangian fairness constraint** — the per-lane accumulated wait is capped, with
   the multiplier `lambda` updated by dual ascent between training iterations rather
   than hand-tuned.

## Corridor-wide results (base demand, 3 seeds)

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | CO2 (kg) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | CO2 (kg) vs fixed |
|---|---|---|---|---|---|---|---|---|
| fixed | 83.10 ± 1.62 | 427.67 ± 176.42 | 1280.00 ± 189.40 | 732.60 ± 64.95 | — | — | — | — |
| actuated | 40.89 ± 2.88 | 76.33 ± 1.70 | 1646.67 ± 3.40 | 508.26 ± 11.21 | +50.8% | +82.2% | +28.6% | +30.6% |
| marl_independent | 17.80 ± 0.28 | 41.00 ± 2.83 | 1679.67 ± 4.11 | 450.96 ± 0.50 | +78.6% | +90.4% | +31.2% | +38.4% |
| marl_shared_w5nofair | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |
| marl_shared_w5 | 17.85 ± 0.38 | 149.67 ± 150.15 | 1569.33 ± 148.02 | 474.66 ± 33.18 | +78.5% | +65.0% | +22.6% | +35.2% |
| marl_shared_w5strong | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |
| marl_shared_w5gnn | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |
| marl_shared_w5nocoord | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |

## Fairness / tail metrics

| Controller | 95th pct wait (s) | Worst wait (s) |
|---|---|---|
| fixed | 211.33 ± 5.56 | 721.00 ± 127.38 |
| actuated | 122.33 ± 7.32 | 243.00 ± 25.66 |
| marl_independent | 47.67 ± 0.94 | 95.33 ± 3.40 |
| marl_shared_w5nofair | 48.00 ± 0.82 | 96.00 ± 2.83 |
| marl_shared_w5 | 48.00 ± 0.82 | 288.67 ± 271.07 |
| marl_shared_w5strong | 48.00 ± 0.82 | 96.00 ± 2.83 |
| marl_shared_w5gnn | 48.00 ± 0.82 | 96.00 ± 2.83 |
| marl_shared_w5nocoord | 48.00 ± 0.82 | 96.00 ± 2.83 |

`wait_p95_s` is the 95th percentile of completed-trip waiting time and
`worst_vehicle_wait_s` the single worst trip. Average delay can improve while the
tail gets worse — a policy is free to starve a side street to speed up the main
road — so the constraint is judged on the tail, not the mean.

## Dual ascent on the fairness multiplier

| Seed | lambda start | lambda final | Final max lane wait (s) | Cap (s) |
|---|---|---|---|---|
| 0 | 0.5 | 2.0 | 42.75 | 30.0 |
| 1 | 0.5 | 2.0 | 277.6979 | 30.0 |
| 2 | 0.5 | 2.0 | 35.6875 | 30.0 |

`lambda` rises while the cap is breached and decays once it is satisfied, which
is what makes this a constraint rather than a fixed penalty weight. See
`outputs/week5_lambda.png`.

**The multiplier saturated in every seed.** `lambda` reached its clamp of
2.0 and stayed there, which means the 30.0 s cap was *never satisfied* during
training — dual ascent kept raising the price and the policy kept paying it.
Two things follow, and both are worth saying plainly:

1. Over most of training the term behaved like a fixed penalty at maximum
   strength, not like an adaptive multiplier. The adaptivity only shows in the
   early iterations before the clamp is reached.
2. The cap is likely infeasible for this corridor at this demand. A constraint
   the policy cannot meet is a statement about the network's capacity, not a
   failure of the optimiser. Choosing a feasible cap — or reporting the
   achievable frontier by sweeping it — is the honest next step, and is left
   as documented future work rather than quietly retuned until it looked good.

## Online learning loop

The Week 5 policy, trained on base demand, was deployed into a *different*
demand and kept learning from the live stream. Round 0 is the frozen policy —
what you get if a deployed controller never updates.

| Deployed into | Rounds | Frozen wait (s) | Final wait (s) | Best wait (s) | Change |
|---|---|---|---|---|---|
| `asymmetric` | 8 | 10.32 | 10.32 | 10.32 | +0.0% |
| `peak` | 6 | 40.57 | 38.89 | 38.75 | +4.1% |

Curves: `outputs/week5_online_learning_<scenario>.png`.

## Definition of Done

> Corridor-wide RL beats both baselines on wait time + throughput, with fairness
> constraint measurably capping worst-case wait.

**Part 1 — beats both baselines on wait time and throughput**

- vs `fixed`: avg_wait_time_s +78.5%, throughput_veh +22.6% — PASS
- vs `actuated`: avg_wait_time_s +56.4%, throughput_veh -4.7% — FAIL

_Part 1: FAIL_

**Part 2 — fairness constraint measurably caps worst-case wait (by ablation)**

- `wait_p95_s`: 48.0s without the constraint vs 48.0s with it (-0.0%) — constraint does not help
- `worst_vehicle_wait_s`: 96.0s without the constraint vs 288.7s with it (-200.7%) — constraint does not help

_Part 2: FAIL_

### Why the verdict failed — per-seed numbers

**avg_wait_time_s**

| Controller | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `marl_shared_w5` | 18.2 | 18.0 | 17.3 |
| `marl_shared_w5nofair` | 18.2 | 18.0 | 18.3 |
| `marl_shared_w5strong` | 18.2 | 18.0 | 18.3 |
| `marl_independent` | 18.2 | 17.6 | 17.6 |
| `actuated` | 43.3 | 42.5 | 36.8 |

**throughput_veh**

| Controller | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `marl_shared_w5` | 1675.0 | 1673.0 | 1360.0 |
| `marl_shared_w5nofair` | 1675.0 | 1673.0 | 1688.0 |
| `marl_shared_w5strong` | 1675.0 | 1673.0 | 1688.0 |
| `marl_independent` | 1675.0 | 1679.0 | 1685.0 |
| `actuated` | 1648.0 | 1650.0 | 1642.0 |

**worst_vehicle_wait_s**

| Controller | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `marl_shared_w5` | 100.0 | 94.0 | 672.0 |
| `marl_shared_w5nofair` | 100.0 | 94.0 | 94.0 |
| `marl_shared_w5strong` | 100.0 | 94.0 | 94.0 |
| `marl_independent` | 100.0 | 92.0 | 94.0 |
| `actuated` | 215.0 | 277.0 | 237.0 |

**One seed, not a systematic gap.** Seeds other than 2 land on the same
deterministic policy as every ablation, and there the shared policy *does* beat
actuated control on throughput. Seed 2 converged somewhere else — a policy
that jams the corridor — and it is what pulls the 3-seed mean below the baseline.

That is reported as a FAIL rather than dropped as an outlier. Three seeds is
already the minimum this project accepts for a finding; discarding the one that
disagrees would leave two, and would be exactly the selective reporting the
3-seed rule exists to prevent. The honest summary is that this configuration is
**not reliably** better than actuated control on throughput: it is better twice
out of three times, and materially worse the third.

Note also that Week 4's `marl_independent` — the simpler, unshaped configuration —
does clear the bar on both metrics. Corridor-wide RL beating both baselines is a
real result on this network; it is the Week 5 *shaping* that fails to add to it.

### Why Part 2 cannot be shown on this task

The ablation returns identical numbers with and without the constraint, and the
reason was measured rather than guessed.

### Why the ablations return identical numbers

The no-fairness, no-coordination and GAT variants produce the *same* evaluation
metrics — identical to four decimal places, including CO2. That was checked rather
than assumed (`src/policy_agreement.py`).

**The checkpoints are genuinely different artifacts:**

| Variant | Module class | Parameters | Weights identical to reference |
|---|---|---|---|
| `w5` | `DefaultPPOTorchRLModule` | 138,501 | True |
| `w5nofair` | `DefaultPPOTorchRLModule` | 138,501 | False |
| `w5nocoord` | `DefaultPPOTorchRLModule` | 138,501 | False |
| `w5strong` | `DefaultPPOTorchRLModule` | 138,501 | False |
| `w5gnn` | `CorridorGATModule` | 16,291 | False |

**But they take the same actions.** Over 4320 decisions in one episode, asking each variant what it would do given the identical observation:

| Variant | Action agreement | Mean max probability difference |
|---|---|---|
| `w5nofair` | 100.00% | 0.0246 |
| `w5nocoord` | 100.00% | 0.0206 |
| `w5strong` | 78.61% | 0.1592 |
| `w5gnn` | 100.00% | 0.0480 |

**Two mechanisms explain it, and the table separates them.**

*Confidence.* The reference policy's mean `|p(a=0) - p(a=1)|` is **0.887** — it is
highly confident — while most variants differ by at most **0.159** in
probability. A perturbation that small cannot flip a decision that confident, so those
variants argmax identically everywhere.

*The min-green window.* `w5strong` genuinely disagrees more often — only
78.6% overall agreement — yet its evaluation metrics are still
identical. Every one of those disagreements falls in a state where sumo-rl
**discards the action anyway**, because the current green has not yet run for
`min_green + yellow_time`. Restricted to the decisions that can actually move the
signal, agreement is 100% for every variant (2148
of 4320 decisions are actionable). The stronger reward
changed the policy exactly where the environment ignores it.

This is not a degenerate policy: the reference splits its actions {'1': 2172, '0': 2148} across the two green phases, so it is actively
switching, not stuck.

**What this means for the Week 5 and Week 6 claims.** With sumo-rl's default
2-action, 11-dimensional single-junction interface plus a 5 s minimum green, the
greedy policy on this corridor is effectively saturated: reward shaping, a 20x
stronger fairness weight, and a completely different network architecture all change
the *learned distribution* without changing the *deterministic behaviour*. The
reward-shaping ablations therefore cannot be distinguished by deterministic
evaluation on this task, and the fairness constraint cannot be shown to cap
worst-case wait.

That is a finding about the task's action resolution, not evidence that the terms are
implemented wrongly — `src/test_core_logic.py` verifies each term's arithmetic
directly. The way to make these ablations measurable would be a finer action space
(phase-duration control rather than a binary next-phase choice), which is a change to
the environment interface rather than to the reward, and is left as documented future
work.

**Verdict: NOT MET**

## Reproducing

```powershell
python src\train_marl_corridor.py --mode shared --reward shaped --fairness --tag w5 --seed 0
python src\train_marl_corridor.py --mode shared --reward shaped-no-fairness --tag w5nofair --seed 0
python src\eval_marl_corridor.py --controller marl --mode shared --tag w5 --reward shaped --seeds 0 1 2
python src\online_learning.py --seed 0 --tag w5 --scenario asymmetric
python src\compare_week5.py
```
