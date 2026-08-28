# Week 9 — Deferred and Substituted Items

All three Week 9 components ran. Two substitutions were made, both forced by
this machine having no CUDA device, and both recorded here rather than folded
quietly into the result.

## Substituted

| Roadmap item | What was done instead | Why |
|---|---|---|
| LoRA fine-tune of **Phi-3-mini** | LoRA on a much smaller causal model (`week9_config.LORA_BASE_MODEL`) | 3.8B parameters will not fine-tune on a CPU-only machine in any useful time, and the weights alone are a ~7.6 GB download |
| Flower **simulated as multiple processes** | Rounds driven in one process; aggregation is Flower's real `FedAvg` | Flower's simulation runner places clients in Ray actors, and each client here spawns its own SUMO subprocess — the same combination that cost this project a day to Ray oversubscription in Week 4 |

**What the LoRA substitution costs the result.** What is demonstrated is the
adapter pipeline and a measured change against the *identical* base model on a
held-out split. Nothing here is a result about Phi-3-mini, and no number from
this week should be quoted as one. This is fallback item 2 in CLAUDE.md's
priority list, taken deliberately.

**What the Flower substitution costs the result.** Nothing about the algorithm:
the averaging calls `flwr.server.strategy.aggregate.aggregate`, the same code
Flower's own `FedAvg` strategy calls, with the same sample-count weighting
(verified in `test_week9.py`). What is not demonstrated is Flower's transport
layer — client/server messaging, serialisation, failure handling. That is
orchestration, not federated learning.

## Genuinely deferred

- **Emission-smoothing effect on a trained policy.** The reward term is
  implemented and unit-tested — emitting traffic scores strictly lower than
  clean traffic — but no corridor-wide policy has been retrained with it and
  shown to reduce CO2. Doing so is another full Week 5-scale training sweep.
- **Federation across heterogeneous junctions.** Only the four interior
  four-way junctions share an observation and action shape, so only they can
  federate. Averaging across the three-way perimeter junctions would need a
  shape-agnostic scheme — the padded `ObservationAligner` from Week 4, or per-
  shape parameter groups. Neither was attempted.
- **More than three federated clients.** Three districts is the number of
  same-shape junctions available once one is held out. A larger corridor would
  be needed to test whether the effect scales with client count.

## Not deferred, contrary to the earlier version of this file

The previous version of this note listed the emissions reward as "scaffolded in
`src/reward_wrappers.py`, but not integrated into training". The term is now
implemented inside `ShapedReward` itself, where the training loop actually reads
it, and is covered by `test_emissions_term_penalises_emitting_traffic`. Its
*effect on a trained policy* remains unmeasured, which is the deferred item
above — a narrower and more accurate claim than the old one.
