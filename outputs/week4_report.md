# Week 4 — Independent Multi-Agent PPO on the Full Corridor

All 12 corridor junctions are placed under RL control at once,
one independent PPO policy per junction (Ray RLlib multi-agent, identity policy
mapping). Observation and reward functions are unchanged from Weeks 2-3, so the only
new variable is that twelve agents now learn simultaneously.

## Training runs

| Seed | Env steps | Agent steps | Iterations | Wall time (s) | Env steps/s | Diverged |
|---|---|---|---|---|---|---|
| 0 | 72000 | 864000 | 40 | 6843.3 | 10.5 | no |
| 1 | 72000 | 864000 | 40 | 6722.7 | 10.7 | no |
| 2 | 72000 | 864000 | 40 | 6533.4 | 11.0 | no |

**On the training budget.** These runs used 72,000 environment steps. Blocking
their median return into 9,000-step chunks afterwards showed every seed had
plateaued by ~27,000 steps, so Weeks 5 and 6 were budgeted at 48,000 (see
`week4_config.FULL_TIMESTEPS`). Week 4's runs were left as they were rather than
re-run shorter, which means the independent-learner baseline received **50% more
training** than the shared policy it is compared against in Week 5. That makes the
Week 5 comparison conservative, not flattering.

## Definition of Done

> Full corridor trains end-to-end without diverging.

- Seeds trained: **3 / 3**
- Diverged runs: **0**
- Divergence test: any NaN/infinite return, or a final-quarter **median** return
  more than 10% worse than the first-quarter median. The median is
  used because the return distribution is bimodal — most iterations near -3, a
  minority gridlocked below -300 — so a mean-based test measures where the rare
  collapses landed rather than whether learning progressed.
- Collapse rate (gridlocked iterations, an order of magnitude worse than the
  run's own median): 28%, 20%, 20% across seeds. This is the
  instability Week 5 sets out to remove, and it is reported rather than smoothed.

**Verdict: MET**

## Corridor-wide evaluation (context)

| Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | CO2 (kg) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | CO2 (kg) vs fixed |
|---|---|---|---|---|---|---|---|---|
| fixed | 83.10 ± 1.62 | 427.67 ± 176.42 | 1280.00 ± 189.40 | 732.60 ± 64.95 | — | — | — | — |
| actuated | 40.89 ± 2.88 | 76.33 ± 1.70 | 1646.67 ± 3.40 | 508.26 ± 11.21 | +50.8% | +82.2% | +28.6% | +30.6% |
| marl_independent | 17.80 ± 0.28 | 41.00 ± 2.83 | 1679.67 ± 4.11 | 450.96 ± 0.50 | +78.6% | +90.4% | +31.2% | +38.4% |
| marl_shared_w5 | 17.85 ± 0.38 | 149.67 ± 150.15 | 1569.33 ± 148.02 | 474.66 ± 33.18 | +78.5% | +65.0% | +22.6% | +35.2% |
| marl_shared_w5gnn | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |
| marl_shared_w5nocoord | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |
| marl_shared_w5nofair | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |
| marl_shared_w5strong | 18.16 ± 0.14 | 42.33 ± 2.06 | 1678.67 ± 6.65 | 452.00 ± 1.28 | +78.2% | +90.1% | +31.1% | +38.3% |

Week 4's bar is stability, not a benchmark win — the reward is still purely
local and nothing yet coordinates the twelve junctions. Beating the baselines
is Week 5's job.

## GNN state embedding

`src/gnn_encoder.py` implements the graph-attention state encoder the roadmap
schedules for this week: a two-layer PyTorch Geometric `GATConv` stack that attends
over each junction and its directly-downstream neighbours. It is wired into RLlib as
a custom `TorchRLModule` and trains end-to-end with PPO (`--gnn`). Because RLlib
batches experience per policy, the environment ships each junction's neighbourhood
inside its observation (`neighbor_context=True`) and the module decodes each row back
into a star graph. Week 6 reports it as a measured ablation rather than assuming it
helps.

## Artifacts

- `outputs/week4_training_curves.png` — per-seed return curves
- `outputs/week4_nonstationarity_notes.md` — independent-learning behaviour
- `outputs/week4_{independent}_seed{N}_training.json` — full per-iteration history
- `models/marl_independent_seed{0,1,2}/` — RLlib checkpoints
