# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# SmartFlow

Multi-agent digital twin platform for adaptive traffic signal control.
B.Tech CSE final-year major project, CMRCET. Solo build, 12-week window.
OS: Windows. SUMO_HOME must be set as a User environment variable.

## Core Principle — Read This First

Every feature from the original proposal stays in scope. Nothing gets quietly
cut. What changes is HOW each ambitious piece is built — see "De-Risking
Substitutions" below. The MARL core (Phase A–B) is the one thing that must be
rock solid before anything else gets touched. If a later phase runs over
budget, defer in the order listed in "Fallback Priority" below — never
silently drop a feature without documenting it as deferred.

## Tech Stack

- Simulation: SUMO + TraCI (Python)
- RL: Stable-Baselines3 (single-agent) → Ray RLlib (multi-agent), PyTorch
- Backend: FastAPI, one service per domain (sim, rl, vision, graph, llm)
- Frontend: Next.js 15, React, Recharts
- DB: SQLite/CSV in Phase A, Postgres (Supabase free tier) from Phase D
- Graph: Neo4j AuraDB Free · Vector: Chroma · LLM: Claude API
- Containers: Docker → k3d (local Kubernetes) in Phase D

## De-Risking Substitutions (use these, don't ask to swap back)

| Original | Use Instead |
|---|---|
| Cloud Kubernetes (EKS/GKE) | k3d, local single-node cluster |
| Kafka/RabbitMQ | Redpanda (Kafka-API compatible, one container) |
| Neo4j self-hosted | Neo4j AuraDB Free |
| Pinecone/Milvus | Chroma (local) |
| Live camera + YOLOv8 | YOLOv8 trained on UA-DETRAC (public dataset) + rendered SUMO-GUI frames |
| Multi-region federated learning | Flower, simulated as multiple processes on one machine |
| Custom LLM fine-tune | LoRA fine-tune of Phi-3-mini on synthetic Q&A |
| Prometheus+Grafana+ELK | Prometheus+Grafana+Loki, Docker Compose |

## Commands

```powershell
.\venv\Scripts\Activate.ps1
python src\day1_smoketest.py
python src\eval.py --controller fixed --seed 0
python src\eval.py --controller actuated --seed 0
python src\compare_baselines.py
```

## Architecture (Phase A starting point)

```
smartflow/
  configs/    corridor.sumocfg, corridor_actuated.sumocfg
  data/       corridor.net.xml, corridor.rou.xml (the digital twin's inputs)
  src/        eval.py is the benchmark harness — every later phase is judged against it
  outputs/    metrics.csv (append-only log), baseline_comparison.png
```

`eval.py`'s CLI (`--controller`, `--seed`) and the `metrics.csv` column schema
(`controller, seed, avg_wait_time_s, max_queue_len, throughput_veh, total_co2_kg`)
are fixed contracts — every phase from Week 2 onward compares against them.

**SUMO binaries on Windows are not on PATH.** Always construct the path from
`SUMO_HOME`:
```python
SUMO_BIN = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
traci.start([SUMO_BIN, "-c", cfg_path, "--no-step-log", "--no-warnings"])
```

## Non-Negotiable Workflow Rules

- **3-seed minimum** for any RL result before reporting it as a finding.
- **Never let the LLM service write into the RL/sim control loop.** Read-only
  analytics and orchestration only — a hard architectural boundary, not a
  style preference.
- **Verify any SUMO/TraCI/library API call against installed package docs**
  before using it if you're not certain it exists — do not guess method
  names. Run e.g. `python -c "import traci; help(traci.vehicle)"` to check.
- **Commit at the end of every day**, with a message naming what Definition
  of Done was met.
- **Stop and report after each day's Definition of Done is met.** Do not
  continue to the next day or next phase without explicit go-ahead — this
  project is reviewed and explained in a viva, so the person building it
  needs to understand every step, not just receive a finished pile of code.
- **Never claim "production-grade" or "real-time"** in any docstring, README,
  or comment unless it was actually load-tested.
- Before installing a new major dependency not listed in Tech Stack above,
  ask first.

---

# Full 12-Week Roadmap

Read only the section for the phase you're currently in before starting work
— don't try to hold the whole roadmap in context at once.

## Phase A — Foundation & Core Twin (Weeks 1–3)

- **Week 1**: SUMO+TraCI setup, real/grid corridor, eval harness, fixed +
  actuated baselines logged to `outputs/metrics.csv`.
  DoD: `eval.py --controller fixed` and `--controller actuated` both produce
  metrics and `compare_baselines.py` renders a chart.
- **Week 2**: Single-agent PPO (Stable-Baselines3) on a standard benchmark
  intersection, validated against literature before touching the real corridor.
  DoD: trained policy beats fixed-time on the benchmark, 3-seed average.
- **Week 3**: Move validated single-agent pipeline to one real intersection
  from the corridor. Tune state (queue, phase, elapsed time) and reward
  (negative delta in cumulative wait).
  DoD: RL beats both baselines on ≥2 of 3 metrics, 3-seed average + variance.

## Phase B — Multi-Agent Core, Fairness & GNN (Weeks 4–6)

- **Week 4**: Independent PPO per intersection across the full corridor; begin
  GNN state embedding (PyTorch Geometric, graph-attention over neighbors).
  DoD: full corridor trains end-to-end without diverging.
- **Week 5**: Parameter-shared policy, green-wave reward shaping, Lagrangian
  fairness constraint (caps max per-lane wait time), online learning loop.
  DoD: corridor-wide RL beats both baselines on wait time + throughput, with
  fairness constraint measurably capping worst-case wait.
- **Week 6**: Stress-test across light/peak/asymmetric demand. Write the
  internal benchmark report. Deliberately light — absorbs Week 4–5 overflow.
  DoD: viva-defensible benchmark report, including one honest case where RL
  doesn't win.

## Phase C — Feature Sprint (Weeks 7–9)

- **Week 7**: Neo4j AuraDB knowledge graph (roads/sensors/rules) + Chroma RAG;
  LLM Service queries both before answering.
  DoD: a natural-language query returns an answer grounded in real graph
  data, not a hallucinated guess.
- **Week 8**: YOLOv8 on UA-DETRAC + rendered SUMO-GUI frames; statistical
  anomaly detection on the eval harness's live metrics; scenario planner
  (closures, weather perturbations).
  DoD: vision model detects vehicles in a held-out clip; an injected anomaly
  is flagged automatically; one closure scenario completes end-to-end.
- **Week 9**: Flower-based federated learning simulation (multi-process
  "districts"); LoRA fine-tune of Phi-3-mini on synthetic traffic Q&A;
  priority-routing preemption + emission-smoothing reward term.
  DoD: federated averaging measurably improves a held-out district's policy;
  fine-tuned model outperforms base prompt on domain questions.

**This is the densest phase in the plan.** Use the Fallback Priority below
the moment a week starts slipping — don't wait until Week 11 to notice.

## Phase D — Platform & Integration (Weeks 10–11)

- **Week 10**: Split backend into 5 domain FastAPI services communicating via
  Redpanda; Auth0 login; GitHub Actions CI/CD.
  DoD: a push to main rebuilds and redeploys every service automatically,
  behind an authenticated dashboard.
- **Week 11**: Kubernetes manifests + Helm chart, deployed on k3d; Prometheus
  + Grafana + Loki; full Next.js dashboard surface (live runs, KG explorer,
  vision overlay, anomaly alerts, priority-routing toggle, external API docs).
  DoD: `helm install` brings up every service on k3d; Grafana shows live
  dashboards; the UI exposes every feature without a terminal.

## Phase E — Integration, Report & Demo (Week 12)

End-to-end integration test across all 5 services. Final report (architecture,
benchmark results, honest limitations). Verify every citation's DOI directly
against the primary source. Demo video. Mock-viva prep on the RL core and the
fallback decisions actually made.
DoD: a stranger can watch the demo and understand what was built, why each
de-risking choice was made, and what the benchmark proves.

## Fallback Priority — If a Phase Slips, Defer in This Order

Deferred items stay documented as designed-and-scaffolded future work, never
silently dropped.

1. Federated learning simulation (Week 9) — highest complexity, lowest
   dependency from the core MARL result.
2. LoRA fine-tuning (Week 9) — a refinement on an already-working prompted LLM.
3. Full k3d Kubernetes deployment (Week 11) — fall back to Docker Compose,
   same containers.
4. Vision-based monitoring (Week 8) — fall back to documented design without
   a fully trained model.
5. Knowledge graph (Week 7) — fall back to SUMO's own in-memory network graph.

---

# Week 1 — Start Here

This is the active task list. Execute in this exact order. After EACH day
below, stop, show the verification output, and wait for explicit go-ahead
before starting the next day. Do not skip ahead even if confident the next
step will work.

GOAL OF THE WEEK: by Day 7, `python src\eval.py --controller fixed --seed 0`
and `python src\eval.py --controller actuated --seed 0` both run cleanly and
`python src\compare_baselines.py` produces outputs/baseline_comparison.png
comparing them.

### DAY 1 — Environment
- Confirm SUMO is installed and SUMO_HOME is set (ask the user to confirm if
  you can't verify it directly; don't assume).
- Scaffold the repo: configs/, data/, src/, outputs/ folders.
- Create a venv, install sumolib, traci, pandas, matplotlib.
- Write src/day1_smoketest.py: a minimal TraCI script that builds a tiny
  generated network + one vehicle, steps the simulation 20 times, and prints
  the vehicle's speed each step. Use this to PROVE TraCI can launch SUMO and
  read live state — verify API calls against installed package docs if
  uncertain, don't assume.
- Run it. Show the output. DoD: it prints SUCCESS with no errors.

### DAY 2 — Real corridor
- Ask which city/area to use, or suggest OpenStreetMap's osmWebWizard for a
  3–8 intersection corridor.
- Walk the user through running osmWebWizard.py interactively (they do the
  browser part — you can't drive a browser map selection for them).
- Once told the files are downloaded, write the commands to move them into
  data/ and decompress the network file to plain corridor.net.xml (not
  gzipped — it gets hand-edited later, keep it plain XML from the start).
- If the import looks broken, fall back to generateNet.py for a 4x4 synthetic
  grid instead — don't let this burn the whole day.
- Verify by opening it in sumo-gui. DoD: the real network renders visibly.

### DAY 3 — Demand
- Generate traffic demand with randomTrips.py (or use osmWebWizard's
  generated routes if those exist and look reasonable).
- Write configs/corridor.sumocfg tying the network and routes together,
  1800 simulated seconds.
- Verify visually in sumo-gui — cars should be moving through the corridor.
  DoD: visible traffic flow, not an empty network.

### DAY 4 — Eval harness (the most important file this week)
- Write src/eval.py: takes --controller {fixed,actuated} and --seed, runs
  the scenario via TraCI, and logs avg_wait_time_s, max_queue_len,
  throughput_veh, total_co2_kg to outputs/metrics.csv (append mode, write
  header only if the file doesn't exist yet).
- Use traci.vehicle.getWaitingTime, traci.vehicle.getCO2Emission,
  traci.lane.getLastStepHaltingNumber, and traci.simulation.getArrivedNumber
  — verify these exist in the installed traci package before relying on them.
- Test first with a short 120-second run so bugs surface fast. Once clean,
  restore the full 1800-second run.
  DoD: one clean row appears in outputs/metrics.csv.

### DAY 5 — Fixed-time baseline
- Run eval.py --controller fixed across seeds 0, 1, 2.
- Show the 3 rows. Flag it explicitly if metrics vary wildly across seeds
  rather than silently proceeding — that would mean demand generation has a
  randomness problem worth fixing before Week 2.
  DoD: 3 stable fixed-time rows logged.

### DAY 6 — Actuated baseline + comparison chart
- Create data/corridor_actuated.net.xml: same network, every tlLogic's type
  attribute changed from static to actuated. Do this with a script, not by
  hand-editing, so it's reproducible.
- Create configs/corridor_actuated.sumocfg pointing at it.
- Update eval.py so --controller actuated loads the actuated config.
- Run eval.py --controller actuated across seeds 0, 1, 2.
- Write src/compare_baselines.py: read outputs/metrics.csv, average by
  controller, plot a 3-panel bar chart (avg wait, max queue, throughput),
  save to outputs/baseline_comparison.png.
  DoD: the chart exists with real numbers from the corridor — report honestly
  if actuated doesn't beat fixed-time, that's still a valid Week 1 result.

### DAY 7 — Buffer, README, commit
- Close any gaps from Days 1–6 — don't start anything new today.
- Write README.md documenting setup, commands, file map, and a results table
  (real numbers from outputs/metrics.csv).
- Write .gitignore (venv/, __pycache__/, *.pyc — but NOT outputs/, since the
  chart and CSV are the proof of work for this portfolio project).
- git init, add, commit with a message describing what Week 1 proves.
- Ask before pushing to a remote — confirm the GitHub repo URL first.

Throughout: if about to write a SUMO/TraCI API call that isn't certain to
exist, check it against the installed package rather than guessing. Never
claim something works without having actually run it and shown the output.

---

## After Week 1 Is Done

Start each new week in a **fresh Claude Code session** — a clean context
with the right prompt beats a long session full of accumulated back-and-forth.
Use `next_week_template.txt` as the starting prompt template (fill in week
number, phase, and prior-week numbers from `outputs/metrics.csv`). Before
writing any code, skim `outputs/metrics.csv` for the real numbers from prior
weeks — never assume values from memory.

Do not invent Week 2's tasks from memory — re-read the "Phase A" section
above (Week 2 entry) and ask the user to confirm before starting it, since
each week should be a deliberate go-ahead, not an automatic continuation.
