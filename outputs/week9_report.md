# Week 9 — Federated Learning, LoRA Fine-Tuning and Priority Routing

Three components. Two carry the week's stated Definitions of Done; the third — priority routing and the emissions reward term — is a deliverable without a numeric DoD, and is measured anyway. Every number below is read from the recorded artifacts by `src/week9_report.py`.

## 1. Federated learning across districts

> **DoD** — federated averaging measurably improves a held-out district's policy.

**Verdict: NOT MET.** Districts `B1`, `B2`, `C1` each trained a policy on their own junction only, over 3 rounds of 3,000 steps, averaged with Flower's FedAvg after every round. Evaluated on junction **C2**, which took no part in training, across 3 seeds:

| Policy | Wait at held-out junction (s) |
|---|---:|
| fixed-time program | 22.32 |
| mean of the local policies | 5.84 |
| **federated average** | **5.84** |

Federated averaging comes out **+0.0%** against the local-only baseline. That baseline is the comparison that matters: transferring any one district's policy is what you get *without* federation, so beating the mean of those transfers is the whole claim.

### Why the answer is exactly zero

Federated averaging changes nothing here, and the reason is not that the averaging is broken. The policies are genuinely different objects — mean L2 distance between district weight vectors is **1.41**, so training on different traffic did move them apart — but they **agree on 100.0% of their decisions** on the held-out junction. Averaging distinct weights that induce the same policy produces the same policy again.

This is the **third independent sighting of the same effect**. Week 5 found five shaped variants with byte-identical metrics; Week 6's `policy_agreement.py` showed they agreed on 100% of actionable decisions; Week 9 now finds that separately-trained district policies do too. The cause is the action space, not the algorithm: a binary next-phase action with a 5 s minimum green leaves so little room that almost every reasonable policy collapses onto the same greedy rule.

**So the Week 9 DoD fails for a structural reason, and the failure is informative.** Federated learning cannot demonstrate a benefit on a task where every client already learns the same controller. Testing it properly needs a finer action space — phase-duration control rather than binary next-phase — which is the same fix Week 6 identified and which remains the clearest next piece of work in this project.

What *is* demonstrated, and is worth keeping: the full federated loop runs end to end — local training per district, Flower's FedAvg aggregation with sample-count weighting, and evaluation on a district held out of training entirely. The machinery is correct and unit-tested; the task is what refuses to differentiate.

### Per-seed detail

| Seed | fixed | local mean | fedavg |
|---|---:|---:|---:|
| 0 | 18.33 | 4.84 | 4.84 |
| 1 | 30.80 | 6.05 | 6.05 |
| 2 | 17.84 | 6.62 | 6.62 |

### Two decisions worth defending

- **Only the four interior junctions can federate.** B1, B2, C1 and C2 are the corridor's only four-way junctions and the only ones that observe an 11-feature state with a 2-action space. The eight perimeter junctions are three-way and observe 9 features. Averaging weights across different architectures is meaningless, so they are excluded rather than padded. `test_week9.py` asserts this.
- **Rounds are driven in one process; the aggregation is Flower's.** The roadmap asks for Flower "simulated as multiple processes". Flower's own simulation runner places clients in Ray actors, and each client here spawns its own SUMO subprocess — the same combination that cost this project a day to Ray oversubscription in Week 4. The rounds therefore run in-process while the averaging calls the real `flwr.server.strategy.aggregate.aggregate`. That changes the orchestration, not the algorithm.

### A baseline bug worth recording

The first version of this experiment evaluated the fixed-time baseline by holding action 0 inside the controlled environment, and reported an implausible 0.78 s wait. Holding action 0 does not reproduce the signal program — it pins one approach permanently green. The baseline now runs a plain SUMO episode with nothing overriding the signals.

## 2. LoRA fine-tuning on synthetic traffic Q&A

> **DoD** — the fine-tuned model outperforms the base prompt on domain questions.

**Verdict: MET.**

| Measure | Base | LoRA | Change |
|---|---:|---:|---:|
| held-out perplexity | 13.24 | 1.83 | -86.2% |
| answer accuracy | 0.034 | 0.080 | +0.046 |

The adapter trains **294,912** of **82,207,488** parameters (0.36%) for 12 epochs on CPU.

The corpus is **397 Q&A pairs** generated from the project's own knowledge graph and metrics across 9 fact kinds (degree, lanes, phases, position, program, result, rule, sensors, topology), split 310/87.

### The split is by fact, not by record

Several phrasings are generated per fact. Splitting record-by-record would put paraphrases of the same fact on both sides, and the held-out score would then measure memorisation. Every phrasing of one fact goes to the same side; `test_week9.py` asserts zero group overlap and zero verbatim answer overlap.

### Substitution: not Phi-3-mini

The roadmap names Phi-3-mini. This machine has no CUDA device, and LoRA on 3.8B parameters will not finish on CPU in any useful time, so `distilgpt2` stands in. What is demonstrated is the adapter pipeline and a measured change against the identical base model. It is **not** a result about Phi-3, and nothing here should be quoted as one. This is fallback item 2 in CLAUDE.md's priority list, taken deliberately and recorded rather than dropped.

### Reading these two numbers together

Perplexity says the adapter fits the domain's phrasing; answer accuracy asks whether a greedily decoded answer contains the right fact. They measure different things and can move independently, which is why both are reported.

Here both improved. But the honest reading of the pair is that the collapse in perplexity (-86%) is a much larger effect than the gain in accuracy, which remains low in absolute terms (0.080). The adapter has clearly learned the *shape* of a corridor answer - the vocabulary, the sentence form, the units - and only marginally learned the specific numbers. That is the expected outcome for a model this size trained on a few hundred pairs, and it is the reason perplexity alone would have been a misleading headline.

Neither number should be read as a claim that this model is a useful traffic assistant. The grounded question-answering that actually works in this project is Week 7's retrieval service, which answers from the graph and the reports rather than from weights.

## 3. Priority routing and emission smoothing

These are Week 9 deliverables without a numeric DoD of their own. Both are implemented and measured.

### Emergency-vehicle preemption

12 emergency vehicles per episode, 3 seeds. A signal is preempted when an emergency vehicle is within 120 m of it on an incoming lane.

| Measure | No preemption | With preemption | Change |
|---|---:|---:|---:|
| emergency vehicle wait (s) | 416.39 | 58.22 | −86% |
| general traffic wait (s) | 78.43 | 77.93 | -0.6% |

**Preemption costs general traffic nothing here — it slightly helps it.** That is not the expected textbook result and deserves an explanation rather than a victory lap. On a corridor this close to saturation, an emergency vehicle stopped at a red is itself an obstruction: it holds a lane and the queue behind it propagates. Clearing it early removes that blockage, and throughput rises in every seed. On a lightly loaded network the trade would almost certainly go the other way, and this result should not be generalised beyond the demand it was measured at.

### Emission smoothing

`ShapedReward` had `emissions_beta` declared and documented as "reserved for Week 9" but never computed. The term is now implemented: CO2 per vehicle on the incoming lanes, converted to grams so the weight sits on the same scale as the delay term.

It is deliberately **per vehicle, not total**. A total rewards emptying the junction, which the delay term already does, and would double-count it. The default weight is small (0.05) for the same reason: CO2 and waiting time are strongly correlated on this network, so a large weight mostly re-scales the delay term and teaches nothing new. `test_week9.py` verifies that emitting traffic scores strictly lower than clean traffic with the term enabled.

**What is not claimed:** no corridor-wide policy has been retrained with the emissions term and shown to reduce CO2. The term is implemented and unit-tested; its effect on a trained policy is unmeasured and is left as future work rather than asserted.

## Week 9 summary

| Component | DoD | Verdict |
|---|---|---|
| Federated learning | averaging improves a held-out district | **NOT MET** |
| LoRA fine-tune | beats the base model on domain questions | **MET** |
| Priority routing | no numeric DoD; measured | implemented and measured |
| Emission smoothing | no numeric DoD; implemented | implemented, unit-tested |

The federated DoD is recorded as **NOT MET**. It fails because the task saturates, not because the federation is broken: separately trained district policies converge on the same controller, so averaging them has nothing to improve. That is the same argmax-saturation effect Weeks 5 and 6 documented, now confirmed a third way, and it is reported as a failure rather than re-scoped until it passed.

Verification: `python src/test_week9.py` — 9 checks covering FedAvg's weighting, policy weight round-tripping, Q&A split integrity, corpus fidelity against the graph, the emissions term's sign, and the shape constraint on which junctions may federate.
