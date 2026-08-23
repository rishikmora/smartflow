# SmartFlow Internal Benchmark Report (Phase A-B)

Covers Weeks 1-6: fixed-time and actuated baselines, single-agent PPO on one
junction, independent multi-agent PPO across the corridor, and the parameter-shared
policy with green-wave shaping and a Lagrangian fairness constraint. Every number is
a 3-seed mean ± standard deviation, measured by the same per-second metric collector
(`src/metrics.py`) for RL and baselines alike.

## Network and demand

`data/corridor.net.xml` — a 12-junction signalised grid (4x4 minus corners). Episodes are 1800 simulated seconds; agents decide every 5 s
with a 2 s yellow, 5 s minimum green and 90 s maximum green.

| Scenario | Vehicles | Description |
|---|---|---|
| `base` | 1800 | the demand every policy was trained on (1800 vehicles / 1800 s) |
| `light` | 900 | ~50% of base demand |
| `peak` | 2700 | ~150% of base demand |
| `asymmetric` | 1397 | full east-west demand, north-south thinned to 40% |

All policies were trained on `base` only. `light`, `peak` and `asymmetric` are
held-out demand, so those columns measure generalisation, not fit.

## Results by scenario

### `base` — the demand every policy was trained on (1800 vehicles / 1800 s)

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | 95th pct wait (s) | Worst wait (s) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | 95th pct wait (s) vs fixed | Worst wait (s) vs fixed |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 83.10 ± 1.62 | 427.67 ± 176.42 | 1280.00 ± 189.40 | 211.33 ± 5.56 | 721.00 ± 127.38 | — | — | — | — | — |
| actuated | 40.89 ± 2.88 | 76.33 ± 1.70 | 1646.67 ± 3.40 | 122.33 ± 7.32 | 243.00 ± 25.66 | +50.8% | +82.2% | +28.6% | +42.1% | +66.3% |
| marl_independent | 17.80 ± 0.28 | 41.00 ± 2.83 | 1679.67 ± 4.11 | 47.67 ± 0.94 | 95.33 ± 3.40 | +78.6% | +90.4% | +31.2% | +77.4% | +86.8% |
| marl_shared_w5 | 17.85 ± 0.38 | 149.67 ± 150.15 | 1569.33 ± 148.02 | 48.00 ± 0.82 | 288.67 ± 271.07 | +78.5% | +65.0% | +22.6% | +77.3% | +60.0% |
| marl_shared_w5gnn | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5nocoord | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5nofair | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5strong | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |

### `light` — ~50% of base demand

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | 95th pct wait (s) | Worst wait (s) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | 95th pct wait (s) vs fixed | Worst wait (s) vs fixed |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 36.99 ± 1.24 | 47.00 ± 1.63 | 836.33 ± 0.47 | 89.67 ± 4.92 | 157.33 ± 10.34 | — | — | — | — | — |
| actuated | 6.35 ± 0.07 | 11.33 ± 0.47 | 859.33 ± 1.25 | 19.00 ± 0.82 | 49.67 ± 9.57 | +82.8% | +75.9% | +2.8% | +78.8% | +68.4% |
| marl_independent | 6.58 ± 0.09 | 16.00 ± 0.82 | 855.00 ± 1.63 | 18.67 ± 0.94 | 47.00 ± 8.83 | +82.2% | +66.0% | +2.2% | +79.2% | +70.1% |
| marl_shared_w5 | 6.70 ± 0.20 | 16.00 ± 0.82 | 855.33 ± 1.25 | 18.33 ± 0.47 | 45.67 ± 7.59 | +81.9% | +66.0% | +2.3% | +79.5% | +71.0% |

### `peak` — ~150% of base demand

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | 95th pct wait (s) | Worst wait (s) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | 95th pct wait (s) vs fixed | Worst wait (s) vs fixed |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 68.87 ± 6.38 | 1158.33 ± 1.25 | 279.00 ± 9.42 | 183.67 ± 26.85 | 1653.33 ± 23.75 | — | — | — | — | — |
| actuated | 71.19 ± 11.17 | 1144.33 ± 7.93 | 425.33 ± 42.93 | 235.00 ± 46.07 | 1575.33 ± 19.07 | -3.4% | +1.2% | +52.5% | -27.9% | +4.7% |
| marl_independent | 42.51 ± 5.61 | 1139.33 ± 5.91 | 528.33 ± 8.18 | 114.00 ± 14.31 | 1348.67 ± 12.04 | +38.3% | +1.6% | +89.4% | +37.9% | +18.4% |
| marl_shared_w5 | 39.24 ± 3.26 | 1136.33 ± 1.25 | 508.67 ± 34.20 | 105.67 ± 11.15 | 1409.67 ± 58.27 | +43.0% | +1.9% | +82.3% | +42.5% | +14.7% |

### `asymmetric` — full east-west demand, north-south thinned to 40%

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | 95th pct wait (s) | Worst wait (s) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | 95th pct wait (s) vs fixed | Worst wait (s) vs fixed |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 54.15 ± 0.75 | 89.00 ± 2.16 | 1269.67 ± 6.34 | 125.00 ± 4.55 | 207.33 ± 14.52 | — | — | — | — | — |
| actuated | 11.32 ± 0.66 | 22.33 ± 1.25 | 1320.00 ± 2.16 | 34.00 ± 2.83 | 79.00 ± 9.90 | +79.1% | +74.9% | +4.0% | +72.8% | +61.9% |
| marl_independent | 10.86 ± 0.48 | 27.00 ± 2.94 | 1320.33 ± 0.47 | 30.33 ± 2.06 | 67.33 ± 5.44 | +79.9% | +69.7% | +4.0% | +75.7% | +67.5% |
| marl_shared_w5 | 10.76 ± 0.34 | 26.00 ± 1.63 | 1322.67 ± 1.89 | 30.33 ± 2.06 | 75.67 ± 14.01 | +80.1% | +70.8% | +4.2% | +75.7% | +63.5% |

## Ablations (base demand)

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | 95th pct wait (s) | Worst wait (s) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | 95th pct wait (s) vs fixed | Worst wait (s) vs fixed |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 83.10 ± 1.62 | 427.67 ± 176.42 | 1280.00 ± 189.40 | 211.33 ± 5.56 | 721.00 ± 127.38 | — | — | — | — | — |
| actuated | 40.89 ± 2.88 | 76.33 ± 1.70 | 1646.67 ± 3.40 | 122.33 ± 7.32 | 243.00 ± 25.66 | +50.8% | +82.2% | +28.6% | +42.1% | +66.3% |
| marl_independent | 17.80 ± 0.28 | 41.00 ± 2.83 | 1679.67 ± 4.11 | 47.67 ± 0.94 | 95.33 ± 3.40 | +78.6% | +90.4% | +31.2% | +77.4% | +86.8% |
| marl_shared_w5nofair | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5nocoord | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5gnn | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5strong | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 48.00 ± 0.82 | 96.00 ± 2.83 | +78.2% | +90.1% | +31.1% | +77.3% | +86.7% |
| marl_shared_w5 | 17.85 ± 0.38 | 149.67 ± 150.15 | 1569.33 ± 148.02 | 48.00 ± 0.82 | 288.67 ± 271.07 | +78.5% | +65.0% | +22.6% | +77.3% | +60.0% |

- `marl_independent` — Week 4 — independent policies, local reward only
- `marl_shared_w5nofair` — shared policy + green wave, **no** fairness constraint
- `marl_shared_w5nocoord` — shared policy + fairness, **no** green-wave term
- `marl_shared_w5gnn` — shared policy + full shaping, GAT encoder instead of MLP
- `marl_shared_w5strong` — as the headline run but with a 20x stronger fairness weight
- `marl_shared_w5` — Week 5 headline — shared policy, green wave + Lagrangian fairness

## Where RL does not win

These cells were found by enumerating every (scenario, baseline, metric)
combination and keeping the ones the RL policy loses. The list is generated from
the data, so it cannot silently omit an unflattering result.

| Scenario | Metric | RL | Baseline | Baseline value | Gap |
|---|---|---|---|---|---|
| `base` | Max queue (veh) | 149.67 | `actuated` | 76.33 | -96.1% |
| `light` | Max queue (veh) | 16.00 | `actuated` | 11.33 | -41.2% |
| `base` | Worst wait (s) | 288.67 | `actuated` | 243.00 | -18.8% |
| `asymmetric` | Max queue (veh) | 26.00 | `actuated` | 22.33 | -16.4% |
| `light` | Avg wait (s) | 6.70 | `actuated` | 6.35 | -5.5% |
| `base` | Throughput (veh) | 1569.33 | `actuated` | 1646.67 | -4.7% |
| `light` | Throughput (veh) | 855.33 | `actuated` | 859.33 | -0.5% |

**The honest failure case.** The clearest loss is `base` demand on
Max queue (veh), where the policy is 96.1% worse than `actuated`.

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

## Methodology checks

Two ways this comparison could have been quietly unfair were checked rather than
assumed away:

- **Same simulator settings for every controller.** sumo-rl starts SUMO with
  `--time-to-teleport -1`, while SUMO's own default teleports a vehicle stuck for
  300 s. Teleporting clears gridlock and would have inflated the baselines'
  throughput relative to the RL runs. Measured on base demand, a fixed-time run
  records **zero teleports** and byte-identical metrics under either setting — but
  `peak` is designed to gridlock, which is where it would bite, so all baselines
  now go through `smartflow_env.baseline_sumo_cmd` and inherit sumo-rl's options.
- **Same sampling rate for every metric.** RL controllers act every 5 simulated
  seconds, so sampling once per decision would divide every waiting time by five.
  The collector is driven once per simulated second for baselines and RL alike
  (`ControlledSumoEnvironment._sumo_step`).

## Honest limitations

- **Single network, single route file.** Every result comes from one 12-junction
  synthetic grid with one randomly generated demand set. Nothing here demonstrates
  transfer to a real road network; the OSM corridor remains deferred from Week 1.
- **Three seeds.** Enough to show variance, not enough for confidence intervals.
  The Week 1 fixed-time baseline already showed high seed variance near saturation.
- **Policies were trained on `base` demand only.** The other scenarios test
  generalisation, and a policy retrained per scenario would very likely do better.
- **`peak` drives the corridor past capacity.** At 150% demand the fixed-time
  baseline gridlocks outright, so differences there partly reflect which controller
  degrades more gracefully rather than which controls traffic better.
- **Simulation only.** No claim is made about real-world or real-time performance;
  nothing in this project has been load-tested.

## Definition of Done

> Viva-defensible benchmark report, including one honest case where RL doesn't win.

- Scenarios evaluated: **4 / 4**
- Cases where RL loses to a baseline, enumerated above: **7**

**Verdict: MET**

## Artifacts

- `outputs/marl_metrics.csv` — every corridor-wide evaluation row
- `outputs/week3_corridor_metrics.csv` — Week 3 single-junction rows
- `outputs/week6_scenarios.png`, `outputs/week6_robustness.png`, `outputs/week6_ablations.png`
- `outputs/week{3,4,5}_report.md` — the per-week reports
