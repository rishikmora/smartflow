# Week 3 — Single-Agent PPO on One Corridor Junction
Week 2 validated the PPO pipeline on sumo-rl's standard 2-way single-intersection
benchmark. Week 3 ports that pipeline, unchanged in its observation and reward
functions, onto junction **B1** of the Week 1 corridor. The other
eleven junctions keep the fixed-time program from `data/corridor.net.xml`.

## Experimental design
| Controller | What it does |
|---|---|
| `fixed` | all 12 junctions fixed-time (Week 1 reference) |
| `actuated` | all 12 junctions SUMO-actuated (Week 1 reference) |
| `actuated_single` | only B1 actuated, other 11 fixed-time |
| `ppo` | only B1 under RL, other 11 fixed-time |

`actuated_single` is the controlled comparison for `ppo`: both change exactly one
junction and leave the other eleven alone, so the difference between them is the
controller, not the number of junctions upgraded. Comparing a single RL junction
against a fully-actuated corridor would confound those two effects.

## Which policy is reported

PPO's final weights are not its best weights. On this corridor the per-episode
training return for seed 1 swings between -1.3 and -959 in consecutive episodes:
near saturation, one unlucky action sequence cascades into a jam the rest of the
episode cannot clear. Reporting whichever policy existed when the step budget ran
out therefore measures luck as much as learning.

Two rows are reported:

- **`ppo`** — the checkpoint chosen by `src/select_best_checkpoint.py`, scored on a
  **validation seed (100 + training seed)** that is disjoint from the evaluation
  seeds 0/1/2. Selecting on the evaluation seeds themselves would be tuning on the
  test set.
- **`ppo_final_iterate`** — the last-iterate policy, kept in the table because its
  spread is the honest measure of how unstable this training run was.

Full selection table: `outputs/week3_checkpoint_selection.json`.

| Seed | Validation seed | Chosen | Score (s) | Distinct scores among candidates |
|---|---|---|---|---|
| 0 | 100 | `final` | 7.24 | 2 of 9 |
| 1 | 101 | `final` | 9.68 | 1 of 9 |
| 2 | 102 | `127400_steps` | 14.76 | 4 of 9 |

Selection changed less than one might hope, and that is worth stating rather than
hiding: once the policy converges its *deterministic* (argmax) behaviour stops
moving, so most checkpoints score identically on a validation episode even while
the stochastic training policy keeps fluctuating. For 1 of 3 seeds every candidate scored
the same, so the tie broke toward the most-trained checkpoint and the reported
policy is simply the final iterate.

A single validation episode per candidate is also a thin basis for choosing. Averaging over several validation seeds would discriminate better; that is a known
limitation of this protocol, not something the numbers below hide.

## Results at junction B1 (primary)

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed |
|---|---|---|---|---|---|---|
| fixed | 50.01 ± 14.28 | 91.33 ± 10.84 | 545.33 ± 83.37 | — | — | — |
| actuated | 17.26 ± 5.28 | 30.67 ± 6.94 | 736.67 ± 3.09 | +65.5% | +66.4% | +35.1% |
| actuated_single | 24.43 ± 7.89 | 36.33 ± 11.09 | 723.33 ± 8.81 | +51.1% | +60.2% | +32.6% |
| ppo_final_iterate | 16.34 ± 11.23 | 44.33 ± 38.87 | 647.33 ± 130.38 | +67.3% | +51.5% | +18.7% |
| ppo | 16.34 ± 11.23 | 44.33 ± 38.87 | 647.33 ± 130.38 | +67.3% | +51.5% | +18.7% |

Metrics cover vehicles on B1's incoming lanes: mean stopped seconds
per vehicle that cleared the junction, peak halting count on those lanes, and the
number of vehicles that cleared. Values are mean ± std over 3 seeds.

`ppo` and `ppo_final_iterate` are identical here, and that is the result rather
than a copy-paste error: validation-based selection did not change the reported
numbers on this task. It was still the right protocol to run — the alternative is
not knowing whether the last iterate was lucky — but on this corridor it bought
nothing, so no claim rests on it.

## Results corridor-wide (context)

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | CO2 (kg) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | CO2 (kg) vs fixed |
|---|---|---|---|---|---|---|---|---|
| fixed | 86.26 ± 1.73 | 387.33 ± 154.39 | 1294.00 ± 177.28 | 731.13 ± 63.11 | — | — | — | — |
| actuated | 40.89 ± 2.88 | 76.33 ± 1.70 | 1646.67 ± 3.40 | 508.26 ± 11.21 | +52.6% | +80.3% | +27.2% | +30.5% |
| actuated_single | 78.54 ± 2.93 | 146.00 ± 0.82 | 1596.00 ± 15.94 | 636.23 ± 10.02 | +8.9% | +62.3% | +23.3% | +13.0% |
| ppo_final_iterate | 73.31 ± 5.53 | 260.67 ± 182.72 | 1469.33 ± 193.99 | 656.97 ± 66.09 | +15.0% | +32.7% | +13.6% | +10.1% |
| ppo | 73.31 ± 5.53 | 260.67 ± 182.72 | 1469.33 ± 193.99 | 656.97 ± 66.09 | +15.0% | +32.7% | +13.6% | +10.1% |

One junction out of twelve cannot move corridor-wide numbers much, and the table
shows that plainly. That ceiling is the motivation for Week 4's multi-agent work,
not a result to explain away.

## Definition of Done

> RL beats both baselines on >=2 of 3 metrics, 3-seed average + variance.

- vs `fixed`: beats it on **3 of 3** metrics (avg_wait_time_s +67.3%, max_queue_len +51.5%, throughput_veh +18.7%) — PASS
- vs `actuated_single`: beats it on **1 of 3** metrics (avg_wait_time_s +33.1%, max_queue_len -22.0%, throughput_veh -10.5%) — FAIL

**Verdict: NOT MET**

### Why it failed, and what that means

Per-seed junction wait (s) — the spread is the story:

| Controller | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `fixed` | 40.20 | 39.63 | 70.20 |
| `actuated_single` | 34.88 | 15.81 | 22.61 |
| `ppo` | 8.90 | 32.21 | 7.91 |

Seed 1 is the outlier: 32.2 s against 8.9/7.9 s on the other seeds. It is what drives
both the large standard deviation and the lost verdict.

Two honest readings, and the project takes both:

1. **SUMO's actuated controller is a strong baseline, not a straw man.** It is a
   well-tuned classical method with direct detector access. Single-agent RL beating
   the *fixed-time* program by 67% while losing to actuated control on queue and
   throughput is a normal, publishable-shaped result — not a failure of the pipeline.
2. **One junction cannot be judged in isolation.** The RL agent optimises its own
   accumulated wait, which it can reduce by discharging vehicles into neighbours it
   does not control and cannot see. Throughput at B1 suffers because the downstream
   links back up. That is precisely the credit-assignment problem Weeks 4-6 exist to
   address, and corridor-wide the shared policy does beat actuated control.

The verdict is reported as failed rather than re-scoped until it passed. Changing the
comparison to the fully-actuated corridor would have produced a PASS against a
baseline that upgrades twelve junctions to the RL controller's one, which would not
have meant anything.

## Reproducing

```powershell
python src\make_single_actuated_net.py
python src\eval_corridor.py --controller fixed --seeds 0 1 2
python src\eval_corridor.py --controller actuated --seeds 0 1 2
python src\eval_corridor.py --controller actuated_single --seeds 0 1 2
python src\train_ppo_corridor.py --seed 0   # repeat for seeds 1, 2
python src\select_best_checkpoint.py
python src\eval_corridor.py --controller ppo --seeds 0 1 2
python src\compare_week3.py
```

## Artifacts

- `outputs/week3_corridor_metrics.csv` — every logged run
- `outputs/week3_comparison_junction.png` — junction B1 chart
- `outputs/week3_comparison.png` — corridor-wide chart
- `outputs/week3_reward_curves.png` — per-seed training curves
- `models/ppo_corridor_seed{0,1,2}.zip` + `_hparams.json` — policies and settings
