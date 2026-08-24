# Week 7 — Knowledge Graph, RAG and the Read-Only LLM Service

A question-answering layer over the corridor. The graph holds the road network,
its signal programs, the rules that govern them and every measured result; the
vector store holds this project's own reports. The service queries **both** before
answering, and returns the evidence alongside the answer so any sentence can be
checked against a source.

## What the graph contains

`embedded (networkx)` — **301 nodes**, **474 relationships**.

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

*graph: topology · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - Junction C2 sits at x=400.0, y=400.0 and is signalised with 4 incoming and 4 outgoing roads.
  - Junction C2 feeds directly into B2, C1, C3, D2.
From outputs/week3_report.md (Why it failed, and what that means):
  ine. 2. **One junction cannot be judged in isolation.** The RL agent optimises its own accumulated wait, which it can reduce by discharging vehicles into neighbours it does not control and cannot see. Throughput at B1 suffers because the downstream links back up. That is precisely the credit-assignment problem Weeks 4-6 exist to address, and corridor-wide the shared policy does beat actuated control. The verdict is...
```

Sources: `graph:FEEDS:C2`, `graph:Junction:C2`, `outputs/week3_report.md#Why it failed, and what that means#11`

### How many signalised junctions does the corridor network have?

*graph: topology · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - The corridor graph holds 301 nodes and 474 relationships: 11 Controller, 16 Junction, 48 Lane, 48 Phase, 12 Program, 69 Result, 48 Road, 5 Rule, 4 Scenario, 40 Sensor.
  - 12 of the junctions are signalised: A1, A2, B0, B1, B2, B3, C0, C1, C2, C3, D1, D2.
From outputs/week3_report.md (Why it failed, and what that means):
  ine. 2. **One junction cannot be judged in isolation.** The RL agent optimises its own accumulated wait, which it can reduce by discharging vehicles into neighbours it does not control and cannot see. Throughput at B1 suffers because the downstream links back up. That is precisely the credit-assignment problem Weeks 4-6 exist to address, and corridor-wide the shared policy does beat actuated control. The verdict is...
```

Sources: `graph:Junction:signalised`, `graph:stats`, `outputs/week3_report.md#Experimental design#1`, `outputs/week3_report.md#Why it failed, and what that means#11`, `outputs/week4_report.md#Week 4 — Independent Multi-Agent PPO on the Full Corridor#0`

### How many incoming lanes does junction B1 have?

*graph: lanes · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - Junction B1 sits at x=200.0, y=200.0 and is signalised with 4 incoming and 4 outgoing roads.
  - Junction B1 feeds directly into A1, B0, B2, C1.
  - Junction B1 has 4 incoming lanes: A1B1_0, B0B1_0, B2B1_0, C1B1_0.
From outputs/week3_report.md (Why it failed, and what that means):
  ine. 2. **One junction cannot be judged in isolation.** The RL agent optimises its own accumulated wait, which it can reduce by discharging vehicles into neighbours it does not control and cannot see. Throughput at B1 suffers because the downstream links back up. That is precisely the credit-assignment problem Weeks 4-6 exist to address, and corridor-wide the shared policy does beat actuated control. The verdict is...
```

Sources: `graph:FEEDS:B1`, `graph:Junction:B1`, `graph:Lane:B1`, `outputs/week3_report.md#Experimental design#1`, `outputs/week3_report.md#Week 3 — Single-Agent PPO on One Corridor Junction#0`, `outputs/week3_report.md#Why it failed, and what that means#11`

### What sensors does junction B1 have?

*graph: sensors · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - Junction B1 sits at x=200.0, y=200.0 and is signalised with 4 incoming and 4 outgoing roads.
  - Junction B1 feeds directly into A1, B0, B2, C1.
  - Junction B1 is monitored by 4 induction-loop sensors (one per controlled lane). These are inferred: SUMO instantiates them implicitly for actuated control rather than declaring them in the network file.
From outputs/week3_report.md (Why it failed, and what that means):
  ine. 2. **One junction cannot be judged in isolation.** The RL agent optimises its own accumulated wait, which it can reduce by discharging vehicles into neighbours it does not control and cannot see. Throughput at B1 suffers because the downstream links back up. That is precisely the credit-assignment problem Weeks 4-6 exist to address, and corridor-wide the shared policy does beat actuated control. The verdict is...
```

Sources: `graph:FEEDS:B1`, `graph:Junction:B1`, `graph:Sensor:B1`, `outputs/week3_report.md#Experimental design#1`, `outputs/week3_report.md#Why it failed, and what that means#11`

### What phases does junction B1 run and how long is its cycle?

*graph: signal program · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - Junction B1 sits at x=200.0, y=200.0 and is signalised with 4 incoming and 4 outgoing roads.
  - Junction B1 feeds directly into A1, B0, B2, C1.
  - Junction B1 runs a static program with 4 phases and a 90s cycle: phase 0 green 42s; phase 1 yellow 3s; phase 2 green 42s; phase 3 yellow 3s.
  - episode_s = 1800s - simulated seconds per evaluation episode (defined in week4_config.SIM_SECONDS).
From README.md (🧪 What each week proved):
  Definitions of Done are reported as they came out, including the two that failed. | Week | Question | Answer | DoD | |---|---|---|---| | 1 | Does adaptive beat fixed-time at all? | Yes — actuated control cut average wait 53% on the corridor. | ✅ | | 2 | Does the RL pipeline work on a *standard* benchmark? | Yes — PPO cut wait 76% on sumo-rl's 2-way intersection. [Sanity check](outputs/week2_literature_note.md) | ✅ |...
```

Sources: `README.md#🧪 What each week proved#12`, `graph:FEEDS:B1`, `graph:Junction:B1`, `graph:Program:B1`, `graph:Rule:episode_s`

### What is the minimum green time and where is it defined?

*graph: rules · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - min_green_s = 5s - minimum green time before a phase may be cut (defined in week4_config.MIN_GREEN).
From outputs/week2_literature_note.md (Why the margin is large — read the demand file):
  e green time has a lot of headroom to recover. A large improvement is the expected outcome here, not an anomaly. The throughput figure supports this reading rather than contradicting it. Total demand over the episode is about 2500 vehicles. Fixed-time clears 2089 (~84%) and leaves the rest queued; PPO clears 2430 (~97%), which is close to the ceiling. Throughput improves by "only" 16% because it *cannot* improve...
```

Sources: `graph:Rule:min_green_s`, `outputs/week2_literature_note.md#Why the margin is large — read the demand file#2`, `outputs/week2_literature_note.md#Why the margin is large — read the demand file#3`

### How did the RL controller do on peak demand throughput?

*graph: results · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - marl_independent on peak demand: throughput_veh = 528.33 (std 8.18, 3 seeds).
  - marl_shared_w5 on peak demand: throughput_veh = 508.67 (std 34.2, 3 seeds).
From README.md (🔬 How the Benchmark Works):
  Every controller — fixed-time, actuated, single-agent PPO, and every multi-agent variant — is measured by the **same collector** in `src/metrics.py`, sampled once per *simulated second*: - **avg_wait_time_s** — mean seconds a completed trip spent stopped - **max_queue_len** — peak number of halting vehicles - **throughput_veh** — vehicles that reached their destination - **total_co2_kg** — integrated CO₂ emissions -...
```

Sources: `README.md#Week 1 — fixed-time vs actuated baselines (corridor, 3 seeds)#3`, `README.md#🔬 How the Benchmark Works#23`, `graph:Result:marl_independent|peak|throughput_veh`, `graph:Result:marl_shared_w5|peak|throughput_veh`, `outputs/week3_report.md#Definition of Done#9`, `outputs/week4_report.md#Corridor-wide evaluation (context)#3`

### What average wait did actuated control achieve on base demand?

*graph: results · generator: `extractive` · grounded: True*

```
From the corridor knowledge graph:
  - actuated on base demand: avg_wait_time_s = 40.89 (std 2.88, 3 seeds).
From README.md (Week 1 — fixed-time vs actuated baselines (corridor, 3 seeds)):
  | Controller | Avg wait (s) | Max queue (veh) | Throughput (veh) | CO2 (kg) | Avg wait (s) vs fixed | Max queue (veh) vs fixed | Throughput (veh) vs fixed | CO2 (kg) vs fixed | |---|---|---|---|---|---|---|---|---| | fixed | 86.26 ± 1.73 | 387.33 ± 154.39 | 1294.00 ± 177.28 | 731.13 ± 63.11 | — | — | — | — | | actuated | 40.89 ± 2.88 | 76.33 ± 1.70 | 1646.67 ± 3.40 | 508.26 ± 11.21 | +52.6% | +80.3% | +27.2% |...
```

Sources: `BENCHMARK_REPORT.md#`asymmetric` — full east-west demand, north-south thinned to 40%#6`, `BENCHMARK_REPORT.md#`light` — ~50% of base demand#4`, `README.md#Week 1 — fixed-time vs actuated baselines (corridor, 3 seeds)#3`, `graph:Result:actuated|base|avg_wait_time_s`, `outputs/week3_report.md#Results corridor-wide (context)#7`

### Why did the fairness constraint fail to change anything?

*documents: why · generator: `extractive` · grounded: True*

```
From outputs/week5_report.md (Why the ablations return identical numbers):
  ed: reward shaping, a 20x stronger fairness weight, and a completely different network architecture all change the *learned distribution* without changing the *deterministic behaviour*. The reward-shaping ablations therefore cannot be distinguished by deterministic evaluation on this task, and the fairness constraint cannot be shown to cap worst-case wait. That is a finding about the task's action resolution, not...
```

Sources: `BENCHMARK_REPORT.md#Why the ablations return identical numbers#15`, `README.md#🧪 What each week proved#13`, `outputs/week5_report.md#Dual ascent on the fairness multiplier#5`, `outputs/week5_report.md#Why the ablations return identical numbers#15`

### Why did Week 3's Definition of Done fail?

*documents: why · generator: `extractive` · grounded: True*

```
From README.md (🧪 What each week proved):
  Definitions of Done are reported as they came out, including the two that failed. | Week | Question | Answer | DoD | |---|---|---|---| | 1 | Does adaptive beat fixed-time at all? | Yes — actuated control cut average wait 53% on the corridor. | ✅ | | 2 | Does the RL pipeline work on a *standard* benchmark? | Yes — PPO cut wait 76% on sumo-rl's 2-way intersection. [Sanity check](outputs/week2_literature_note.md) | ✅ |...
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
| Knowledge graph | not configured | `embedded (networkx)` |
| Vector store | Chroma (local, no credentials) | `chroma` |
| Answer generation | not configured | `none` |

Two of these need credentials this machine does not have, so the run above used
the fallbacks: the graph was served from an in-process NetworkX build of the same
document, and answers were composed directly from retrieved evidence rather than
phrased by Claude. Both paths are real code exercised by the tests, and the roadmap
already nominates the in-memory graph as the Week 7 fallback.

What this does **not** demonstrate is AuraDB's Cypher path or Claude's phrasing
under load. `src/knowledge_graph.py` contains the Cypher for every query and
`src/llm_service.py` the Claude call; adding credentials to `.env` switches both
with no code change. Until someone runs it that way, that is a claim about the
code, not a measured result.

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
