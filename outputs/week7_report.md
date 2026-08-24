# Week 7 — Knowledge Graph, RAG and the Read-Only LLM Service

A question-answering layer over the corridor. The graph holds the road network,
its signal programs, the rules that govern them and every measured result; the
vector store holds this project's own reports. The service queries **both** before
answering, and returns the evidence alongside the answer so any sentence can be
checked against a source.

## What the graph contains

`neo4j aura` — **301 nodes**, **474 relationships**.

| Node type | Count | Derived from |
|---|---|---|
| `Controller` | 11 | controllers present in the metrics CSVs |
| `Junction` | 16 | `corridor.net.xml` nodes |
| `Lane` | 48 | `corridor.net.xml` lanes |
| `Phase` | 48 | phases inside each `tlLogic` |
| `Program` | 12 | `tlLogic` blocks in the network |
| `Result` | 69 | 3-seed means from `marl_metrics.csv` and `week3_corridor_metrics.csv` |
| `Road` | 48 | `corridor.net.xml` edges |
| `Rule` | 5 | signal-timing constants in `week4_config.py` |
| `Scenario` | 4 | demand scenarios present in the metrics CSVs |
| `Sensor` | 40 | inferred: one induction loop per controlled lane of an actuated program |

Nothing here is hand-authored. `src/test_week7.py` re-derives the topology, the
signal programs and every result mean straight from the source files and compares
them, so an ingestion bug cannot become a confidently-cited wrong answer.

**On Sensor nodes.** The network declares no detectors. SUMO instantiates one
induction loop per controlled lane implicitly when a program is actuated, so these
nodes are inferred and carry `inferred: true` plus a note saying so. They are
included because 'sensors' is part of what this week asks the graph to model;
presenting them as declared infrastructure would be the dishonest option.

## Definition of Done

> A natural-language query returns an answer grounded in real graph data,
> not a hallucinated guess.

- Questions asked: **12**
- Grounded answers: **10**, each carrying its citations
- Out-of-domain questions correctly refused: **2 / 2**
- In-domain questions wrongly refused: **0**

**Verdict: MET**

The refusals are the part that matters. Vector search returns its top *k* for any
input, so an out-of-domain question still comes back with four confident-looking
chunks; a service that simply answers from whatever it retrieved would answer
those too. A relevance floor (IDF-weighted coverage of the question's vocabulary,
with unknown words counting against the score) drops evidence that is not about the
question, and when nothing survives the service declines instead of guessing.

## Transcript

### Which junctions does C2 feed into?

*graph: topology · generator: `claude` · grounded: True*

```
C2 feeds directly into four junctions: B2, C1, C3, and D2. It is a signalised junction at x=400.0, y=400.0 with 4 incoming and 4 outgoing roads. The available evidence doesn't say anything further about traffic volumes or conditions on those downstream links.
```

Sources: `graph:FEEDS:C2`, `graph:Junction:C2`, `outputs/week3_report.md#Why it failed, and what that means#11`

### How many signalised junctions does the corridor network have?

*graph: topology · generator: `claude` · grounded: True*

```
The corridor network has 12 signalised junctions — A1, A2, B0, B1, B2, B3, C0, C1, C2, C3, D1 and D2 — out of 16 Junction nodes in the graph. The project reports corroborate this: Week 3's controller definitions cover "all 12 junctions", and Week 4 places "All 12 corridor junctions" under RL control with one PPO policy each.
```

Sources: `graph:Junction:signalised`, `graph:stats`, `outputs/week3_report.md#Experimental design#1`, `outputs/week3_report.md#Why it failed, and what that means#11`, `outputs/week4_report.md#Week 4 — Independent Multi-Agent PPO on the Full Corridor#0`

### How many incoming lanes does junction B1 have?

*graph: lanes · generator: `claude` · grounded: True*

```
Junction B1 has 4 incoming lanes: A1B1_0, B0B1_0, B2B1_0, and C1B1_0. These correspond to its 4 incoming roads, from the neighbouring junctions A1, B0, B2 and C1. B1 is the signalised junction at x=200.0, y=200.0 that was placed under RL control in Week 3.
```

Sources: `graph:FEEDS:B1`, `graph:Junction:B1`, `graph:Lane:B1`, `outputs/week3_report.md#Experimental design#1`, `outputs/week3_report.md#Week 3 — Single-Agent PPO on One Corridor Junction#0`, `outputs/week3_report.md#Why it failed, and what that means#11`

### What sensors does junction B1 have?

*graph: sensors · generator: `claude` · grounded: True*

```
Junction B1 is monitored by 4 induction-loop sensors, one per controlled lane. Note that these are inferred rather than declared: SUMO instantiates them implicitly for actuated control instead of listing them in the network file, so they don't appear explicitly in the network definition. No other sensor types at B1 are present in the available data.
```

Sources: `graph:FEEDS:B1`, `graph:Junction:B1`, `graph:Sensor:B1`, `outputs/week3_report.md#Experimental design#1`, `outputs/week3_report.md#Why it failed, and what that means#11`

### What phases does junction B1 run and how long is its cycle?

*graph: signal program · generator: `claude` · grounded: True*

```
Junction B1 runs a static (fixed-time) program with 4 phases totalling a 90s cycle: phase 0 green 42s, phase 1 yellow 3s, phase 2 green 42s, phase 3 yellow 3s. It is a signalised junction at x=200.0, y=200.0 with 4 incoming and 4 outgoing roads, feeding into A1, B0, B2 and C1. No other timing details for B1 are in the available data.
```

Sources: `README.md#🧪 What each week proved#12`, `graph:FEEDS:B1`, `graph:Junction:B1`, `graph:Program:B1`, `graph:Rule:episode_s`

### What is the minimum green time and where is it defined?

*graph: rules · generator: `claude` · grounded: True*

```
The minimum green time is **5 s** — the minimum a phase must be held before it may be cut. It is defined in `week4_config.MIN_GREEN` (recorded in the knowledge graph as the rule `min_green_s`). The supplied document passages cover demand splits and the fixed-time signal program but do not add further detail on this constraint.
```

Sources: `graph:Rule:min_green_s`, `outputs/week2_literature_note.md#Why the margin is large — read the demand file#2`, `outputs/week2_literature_note.md#Why the margin is large — read the demand file#3`

### How did the RL controller do on peak demand throughput?

*graph: results · generator: `claude` · grounded: True*

```
On the peak demand scenario the evidence covers two multi-agent RL variants: `marl_independent` reached a throughput of 528.33 vehicles (std 8.18, 3 seeds) and `marl_shared_w5` reached 508.67 vehicles (std 34.2, 3 seeds) — so the independent variant was higher and far more consistent across seeds. No peak-demand throughput figures for the fixed-time, actuated, or single-agent controllers are present in the supplied evidence, so I can't say how the RL runs compare to a baseline at peak. The corridor-wide table shown is a different (non-peak) evaluation, where `marl_independent` reached 1679.67 ± 4.11 vehicles versus 1646.67 ± 3.40 for actuated. Which RL variant you mean matters here, since the two differ by roughly 20 vehicles at peak.
```

Sources: `README.md#Week 1 — fixed-time vs actuated baselines (corridor, 3 seeds)#3`, `README.md#🔬 How the Benchmark Works#24`, `graph:Result:marl_independent|peak|throughput_veh`, `graph:Result:marl_shared_w5|peak|throughput_veh`, `outputs/week3_report.md#Definition of Done#9`, `outputs/week4_report.md#Corridor-wide evaluation (context)#3`

### What average wait did actuated control achieve on base demand?

*graph: results · generator: `claude` · grounded: True*

```
On base demand, actuated control achieved an average wait of 40.89 s (± 2.88, 3 seeds). That is a 52.6% improvement over the fixed-time baseline's 86.26 ± 1.73 s. It also reached a max queue of 76.33 ± 1.70 veh, throughput of 1646.67 ± 3.40 veh, and CO2 of 508.26 ± 11.21 kg on the same runs.
```

Sources: `BENCHMARK_REPORT.md#`asymmetric` — full east-west demand, north-south thinned to 40%#6`, `BENCHMARK_REPORT.md#`light` — ~50% of base demand#4`, `README.md#Week 1 — fixed-time vs actuated baselines (corridor, 3 seeds)#3`, `graph:Result:actuated|base|avg_wait_time_s`, `outputs/week3_report.md#Results corridor-wide (context)#7`

### Why did the fairness constraint fail to change anything?

*documents: why · generator: `claude` · grounded: True*

```
Per the project's own reports, the fairness weight (even at 20x strength) changes the *learned distribution* but not the *deterministic behaviour*, so under deterministic evaluation the ablation is indistinguishable and the constraint "cannot be shown to cap worst-case wait." The reports attribute this to the task's action resolution — a binary next-phase choice — not to a bug; `src/test_core_logic.py` verifies each term's arithmetic directly. Separately, the dual-ascent multiplier hits its clamp and then behaves like a fixed weight rather than an adaptive one, with adaptivity visible only in early iterations. The reports also note the cap is likely infeasible for this corridor at this demand, which they frame as a statement about network capacity; the suggested measurable alternative is a finer action space (phase-duration control), documented as future work.
```

Sources: `BENCHMARK_REPORT.md#Why the ablations return identical numbers#15`, `outputs/week5_report.md#Dual ascent on the fairness multiplier#5`, `outputs/week5_report.md#Why the ablations return identical numbers#15`

### Why did Week 3's Definition of Done fail?

*documents: why · generator: `claude` · grounded: True*

```
Week 3 asked whether the RL approach transfers to one junction of a real corridor, and the Definition of Done was marked ❌. Although the RL controller beat fixed-time at that junction by 67%, it **lost to an actuated controller** at the same junction on both queue and throughput. Additionally, controlling one junction out of twelve barely moved the corridor overall. Further detail would be in the linked week3_report.md, which isn't included in the evidence here.
```

Sources: `README.md#🧪 What each week proved#12`

### What is the average rainfall in Hyderabad?

*refusal · generator: `refusal` · grounded: False*

```
I don't have data for that. The corridor knowledge graph and the project's reports contain nothing matching this question, and this service only answers from those two sources.
```

### Who won the 2019 cricket world cup?

*refusal · generator: `refusal` · grounded: False*

```
I don't have data for that. The corridor knowledge graph and the project's reports contain nothing matching this question, and this service only answers from those two sources.
```

## Which backends actually ran

| Layer | Configured | Used in this run |
|---|---|---|
| Knowledge graph | Neo4j AuraDB | `neo4j aura` |
| Vector store | Chroma (local, no credentials) | `chroma` |
| Answer generation | Claude API | `claude-opus-5` |

Both credentialed backends ran for real in this run: every graph fact above was fetched over Cypher from AuraDB, and every answer was phrased by Claude from the retrieved evidence. The NetworkX and extractive fallbacks remain in the code and are still exercised by `src/test_week7.py`, so the service degrades rather than breaks when credentials are absent.

## Read-only boundary

CLAUDE.md: *"Never let the LLM service write into the RL/sim control loop."*

`test_week7.py` imports `llm_service` in a clean subprocess and inspects
`sys.modules`, asserting that `traci`, `libsumo`, `sumo_rl`, `ray`,
`stable_baselines3`, `torch` and every training/eval module are absent. The service's
public surface is `ask` and `close` — there is no method that writes anywhere. The
boundary is verified, not asserted.

## Reproducing

```powershell
python src\knowledge_graph.py    # rebuild the graph from the network + CSVs
python src\vector_store.py       # embed the reports into Chroma
python src\test_week7.py         # 9 checks: fidelity, grounding, refusal, boundary
python src\week7_demo.py         # regenerate this report
python src\llm_service.py "which junctions feed C2?"
```
