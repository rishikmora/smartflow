# SmartFlow — Status Index

One line per week, so nothing in this directory has to be read to find out whether
it describes a measured result or an empty scaffold. Every "MET / NOT MET" below is
copied from that week's own report, not re-judged here.

Last updated: 2026-08-24.

## Phase A–B — the RL core (Weeks 1–6)

| Week | Definition of Done | Verdict | Evidence |
|---|---|---|---|
| 1 | Both baselines produce metrics; comparison chart renders | **MET** | `metrics.csv`, `baseline_comparison.png` |
| 2 | PPO beats fixed-time on the sumo-rl benchmark, 3-seed | **MET** | `week2_benchmark_metrics.csv`, `week2_literature_note.md` |
| 3 | RL beats both baselines on ≥2 of 3 metrics, 3-seed | **NOT MET** | `week3_report.md` |
| 4 | Full corridor trains end-to-end without diverging | **MET** | `week4_report.md` |
| 5 | Corridor RL beats both baselines on wait + throughput, fairness caps worst-case wait | **NOT MET** (both parts) | `week5_report.md` |
| 6 | Viva-defensible benchmark report incl. one honest loss | **MET** | `../BENCHMARK_REPORT.md` |

### The two failures, in one line each

- **Week 3** — single-intersection RL beats fixed-time on all three metrics but
  beats actuated on only one of three. Reported as failed rather than re-scoped.
- **Week 5** — the shared policy beats actuated on wait time (17.9 s vs 40.9 s) but
  **not reliably on throughput**: two seeds of three beat it, the third converged to
  a policy that jams the corridor and drags the mean below baseline. The fairness
  constraint measurably changed nothing. Week 4's simpler unshaped configuration
  does clear the bar — it is the Week 5 *shaping* that fails to add to it.

Both are genuine scientific results and stay in the final report. They are not
outstanding work items.

## Phase C — feature sprint (Weeks 7–9)

| Week | Definition of Done | Verdict | Evidence |
|---|---|---|---|
| 7 | A natural-language query returns an answer grounded in real graph data | **MET** | `week7_report.md`, `week7_qa.json` |
| 8 | Vision, anomaly detection, scenario planner | **NOT STARTED** | `week8_deferred.md` |
| 9 | Federated learning, LoRA fine-tune, priority routing | **NOT STARTED** | `week9_deferred.md` |

Week 7 runs against live backends: Neo4j AuraDB over Cypher, Chroma locally, and
`claude-opus-5` for phrasing. Verify with `python src\check_neo4j.py` and
`python src\week7_demo.py`.

Week 8 is blocked on a manual ~5 GB UA-DETRAC download that requires registration.
Weeks 8 and 9 sit at positions 4 and 1–2 in CLAUDE.md's fallback priority, so they
are the first things to defer if the schedule slips.

## Phase D–E — platform and report (Weeks 10–12)

| Week | Definition of Done | Verdict | Evidence |
|---|---|---|---|
| 10 | Push to main rebuilds and redeploys every service, behind auth | **NOT MET — scaffold only** | `week10_report.md` |
| 11 | `helm install` brings up every service on k3d; Grafana live | **NOT STARTED** | `week11_deferred.md` |
| 12 | End-to-end integration, final report, verified citations, demo | **NOT STARTED** | `week12_deferred.md` |

`services/` contains five FastAPI directories, but **every non-health endpoint is a
hardcoded placeholder** — no container has been built and CI has never run. See
`week10_report.md` for the full gap list.

## Files that are drafts, not results

| File | What it is |
|---|---|
| `FINAL_REPORT.md` | Scaffold draft. Deliberately omits unsupported claims until the runs behind them exist. |
| `VIVA_PREP.md` | Draft answer bank, not a record of anything measured. |
| `citations_verified.md` | Correctly states that no DOI has been checked yet. Week 12 work. |

## Interactive artifact

`outputs/viz/corridor_control_room.html` — a self-contained replay of three real
SUMO runs (fixed, actuated, shared-policy RL) over identical traffic. Rebuild with
`python src\viz_build.py`.
