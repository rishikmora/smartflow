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

**SUMO binaries on Windows are not on PATH.** Always reference them via `SUMO_HOME`:
```python
SUMO_BIN = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
traci.start([SUMO_BIN, "-c", cfg_path, "--no-step-log", "--no-warnings"])
```

**actuated TLS requires explicit `minDur`/`maxDur` on green phases.** Without
them, SUMO defaults both to the static phase duration and the controller
behaves identically to fixed-time. See `src/make_actuated_net.py` for the
pattern (minDur=5, maxDur=90 on phases with duration ≥ 10 s).

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

## Code Quality Standard (production-grade engineering, every week from now on)

"Production-grade" applies to HOW the code is written, not to claims about
what it's proven to handle — those stay governed by the rule below about
never claiming "production-grade" or "real-time" without a load test. A
project can be engineered to a high standard while still being honest that
it hasn't been tested at production scale. Both things are true at once;
neither excuses the other.

Concretely, starting now:

- **Logging, not print().** Use Python's `logging` module (INFO for normal
  progress, WARNING for recoverable issues, ERROR for failures) for any
  script beyond a one-off smoke test. Smoke tests can still use print().
- **Type hints** on every function signature.
- **Docstrings** on every script and function — what it does, what it
  expects, what it returns/produces.
- **Error handling that's actually useful.** Wrap file I/O and
  TraCI/SUMO/SB3 calls that can plausibly fail in try/except with a
  message that says what to check, not just a bare re-raise. Validate
  inputs (e.g. check a file exists before loading it) rather than letting
  a missing file surface as a confusing downstream stack trace.
- **No scattered magic paths or strings.** Centralize file paths, env
  names, and constants in one config module per phase (e.g.
  src/week2_config.py) — every script imports from there.
- **Reproducibility.** `pip freeze > requirements.txt` at the end of every
  week, committed. Every RL run logs its seed and hyperparameters
  alongside its results, not just the metrics.
- **Conventional commit prefixes** going forward: `feat:`, `fix:`, `docs:`,
  `chore:` — reads better in a portfolio repo's history than ad hoc messages.
- **Secrets hygiene, starting now even though nothing needs it yet.** Add
  `.env` to .gitignore today. From Phase C onward (Claude API keys, Auth0
  secrets) credentials go in `.env` via python-dotenv, never hardcoded,
  never committed — set the habit before there's anything sensitive to leak.
- **`.gitignore` precision, not blanket ignoring.** Ignore generated noise
  (tensorboard logs, `__pycache__/`, `venv/`) but keep trained model
  checkpoints and result artifacts (`outputs/`, `models/*.zip`) — those are
  the portfolio evidence, the same logic as Week 1's charts.

What this deliberately does NOT include yet: full CI/CD, Kubernetes, or a
formal test suite. Those are scheduled in Phase D for a reason — pulling
them forward now would be the exact "let infra eat time budgeted for the
RL core" mistake the plan already warns against. Production-grade code
discipline this week means clean, typed, logged, reproducible Python — not
standing up infrastructure ahead of schedule.



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

# Week 2 — Start Here

Week 1 is complete: actuated beat fixed on the 4x4 grid (avg wait −53%, max
queue −80%, throughput +27%). Known carryovers — OSM corridor deferred (grid
is valid for all MARL work), seed variance high on fixed at near-saturation
demand, actuated net needs minDur/maxDur set explicitly (already fixed in
make_actuated_net.py).

This week's job: prove single-agent RL works on a STANDARD, literature-used
benchmark — not the Week 1 grid — before touching the real corridor in Week 3.
Use the `sumo-rl` library (LucasAlegre/sumo-rl, pip-installable, Gymnasium +
PettingZoo + Stable-Baselines3 compatible, cited in multiple published traffic
RL papers). Do not build a custom Gym wrapper from scratch — this library's
2-way single-intersection environment IS the standard benchmark.

Execute in this exact order. After EACH day, stop, show the verification
output, and wait for explicit go-ahead before starting the next day.

GOAL OF THE WEEK: a PPO policy trained on sumo-rl's single-intersection
benchmark beats a fixed-time baseline on that same benchmark, averaged across
3 independently trained seeds — with a written note on how the result
compares to literature-reported behavior for this benchmark.

### DAY 1 — Install and smoke-test the benchmark environment
- In the existing venv: `pip install sumo-rl gymnasium "stable_baselines3[extra]>=2.0.0a9"`.
- `sumo-rl`'s pip package may not include the `experiments/` and `nets/`
  folders from its GitHub repo (those aren't guaranteed part of the PyPI
  distribution). VERIFY this — check the installed package's directory
  (`python -c "import sumo_rl, os; print(os.path.dirname(sumo_rl.__file__))"`)
  for a bundled single-intersection net. If it's not there, `git clone
  https://github.com/LucasAlegre/sumo-rl` into a `vendor/` folder just to
  access its `nets/` and `experiments/` reference files — don't guess a path.
- Once found, write the verified net_file/route_file paths into a single
  src/week2_config.py constants file. Every later script this week (Days
  2–6) imports paths from there — do not re-discover or re-guess the path
  in each script.
- **Use sumo-rl's DEFAULT observation function and DEFAULT reward function**
  (change in cumulative vehicle delay) for everything this week. Do not write
  a custom reward or observation function — the entire point of this week is
  comparability with literature, and a custom reward breaks that comparison.
  Only override defaults if explicitly revisited later, never silently.
- Optional performance tip: setting the environment variable
  `LIBSUMO_AS_TRACI=1` can give up to ~8x faster training on CPU, since
  training this week runs far more simulation steps than Week 1's eval did.
  Caveat: this disables sumo-gui and parallel simulation — use it for Day 4's
  actual training runs, but unset it for Day 1's smoke test if visual
  confirmation via sumo-gui is wanted.
- Write src/week2_smoketest.py: instantiate the env via
  `gym.make('sumo-rl-v0', net_file=..., route_file=..., use_gui=False, num_seconds=3600)`,
  reset it, take ~50 random actions via `env.action_space.sample()`, print
  the observation shape and reward each step.
- DoD: 50 random-action steps complete with no errors, obs shape and reward
  values printed and sane (not NaN, not crashing).

### DAY 2 — Pipeline validation: short PPO run
- Write src/train_ppo_benchmark.py using `from stable_baselines3 import PPO`,
  pointed at the sumo-rl single-intersection env.
- Train for a SHORT budget first (e.g. ~20,000–50,000 timesteps) purely to
  validate the training loop runs end-to-end — this is not the real run.
- DoD: training completes without error; reward trend is visible (doesn't
  need to be good yet, just not flat-broken or NaN).

### DAY 3 — Fixed-time baseline on THIS benchmark
- The single-intersection net already has a static tlLogic program (every
  SUMO net does) — write src/eval_benchmark.py to run that fixed-time program
  for 3 seeds and log avg_wait_time_s, max_queue_len, throughput_veh to
  outputs/week2_benchmark_metrics.csv. Same metric definitions as Week 1's
  eval.py, applied to the benchmark net instead of the grid.
- "3 seeds" here means passing different seeds via Gymnasium's
  `env.reset(seed=N)` (which sets SUMO's internal RNG, affecting stochastic
  elements like departure jitter even with the same route file) — not
  regenerating route files from scratch like Week 1 did. Use seeds 0, 1, 2.
- DoD: 3 fixed-time rows logged for the benchmark intersection.

### DAY 4 — The real PPO training run, 3 seeds
- Now that Day 2 validated the pipeline, train 3 SEPARATE PPO models with 3
  different random seeds.
- Concrete starting budget: ~150,000–300,000 timesteps per seed (a single
  intersection is a small task; this range is a reasonable starting point,
  not a hard rule). Use Day 2's reward curve shape to judge whether more or
  less is needed — if it was still climbing steeply at 50,000 steps, lean
  toward 300,000; if it had already flattened, 150,000 is likely enough.
- HARDWARE CHECK FIRST: if Day 2's short run felt slow on this machine even
  with LIBSUMO_AS_TRACI=1, stop and ask the user before committing to 3 full
  runs at this budget — switching to Colab for just this day is a reasonable
  option and better decided before burning hours on CPU, not after.
- Save each model checkpoint: models/ppo_benchmark_seed0.zip, seed1, seed2.
- DoD: 3 trained model files exist, each with a saved reward-curve log.

### DAY 5 — Evaluate the trained policies
- Load each of the 3 trained models, run a deterministic evaluation rollout
  on the benchmark env — explicitly pass `deterministic=True` to
  `model.predict()`, don't rely on a default (SB3's default has varied by
  algorithm/version) — and log the same 3 metrics to
  outputs/week2_benchmark_metrics.csv (controller="ppo", seed=0/1/2).
- Compare PPO's 3-seed average against Day 3's fixed-time 3-seed average.
  PRIMARY METRIC: avg_wait_time_s — this is what the reward signal directly
  optimizes, so it's the headline claim. max_queue_len and throughput_veh are
  supporting evidence, not required to both improve for the week to succeed.
- DoD: report the real numbers either way. If PPO does NOT beat fixed-time on
  avg_wait_time_s, STOP here and diagnose (reward function sign, training
  budget, action/observation wiring) rather than continuing to Day 6 — do not
  paper over a bad result.

### DAY 6 — Comparison chart + literature sanity-check
- If Day 5 is clean: write/adapt a comparison chart script (same pattern as
  Week 1's compare_baselines.py) for outputs/week2_benchmark_comparison.png.
- Add a short written note: does the reward curve shape and the final
  performance gap fall in a believable range compared to what sumo-rl's own
  experiments or the papers citing it report for this benchmark? This is the
  "reproduce a known result" check — it doesn't need to match exactly, it
  needs to not be wildly implausible.
- If Day 5 was NOT clean: this day is diagnosis and re-training instead —
  note explicitly what was wrong and what was changed.
- DoD: chart exists; literature sanity-check note is written either way.

### DAY 7 — Buffer, write-up, commit
- Close any gaps from Days 1–6.
- Update README.md with Week 2 results and the literature comparison note.
- Update .gitignore: ignore tensorboard log directories and __pycache__;
  do NOT ignore models/*.zip or outputs/ — those are portfolio evidence.
- `pip freeze > requirements.txt`, commit it.
- Commit with a message describing what Week 2 proves (conventional prefix:
  `feat:`).
- Confirm go-ahead before Week 3, which ports this validated pipeline onto
  the real Week 1 grid (not this benchmark net).

Throughout: verify any sumo-rl/Gymnasium/SB3 API call against the installed
package before relying on it — this library's exact method signatures matter
more than usual since it's new to the project this week.

---

## Session Protocol

Start each new week in a **fresh Claude Code session** — clean context beats
accumulated back-and-forth. Before writing any code, skim `outputs/metrics.csv`
and any benchmark chart from the prior week for the real numbers.

## After Week 2 Is Done

Do not invent Week 3's tasks from memory — re-read the "Phase A" section
above (Week 3 entry) and ask the user to confirm before starting it.

