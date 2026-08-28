# SmartFlow — Status Index

One line per week, so nothing in this directory has to be read to find out whether
it describes a measured result or an empty scaffold.

Weeks 3, 4, 5, 6 and 7 each state a verdict in their own report, and those are copied
here verbatim. Weeks 1 and 2 predate that convention and have no verdict line, so their
"MET" is read directly off the committed metrics CSVs — the derivation is given in the
footnote below rather than asserted.

Last updated: 2026-08-28.

## Phase A–B — the RL core (Weeks 1–6)

| Week | Definition of Done | Verdict | Evidence |
|---|---|---|---|
| 1 | Both baselines produce metrics; comparison chart renders | **MET** | `metrics.csv`, `baseline_comparison.png` |
| 2 | PPO beats fixed-time on the sumo-rl benchmark, 3-seed | **MET** | `week2_benchmark_metrics.csv`, `week2_literature_note.md` |
| 3 | RL beats both baselines on ≥2 of 3 metrics, 3-seed | **NOT MET** | `week3_report.md` |
| 4 | Full corridor trains end-to-end without diverging | **MET** | `week4_report.md` |
| 5 | Corridor RL beats both baselines on wait + throughput, fairness caps worst-case wait | **NOT MET** (both parts) | `week5_report.md` |
| 6 | Viva-defensible benchmark report incl. one honest loss | **MET** | `../BENCHMARK_REPORT.md` |

**Weeks 1–2, derived not quoted.** Week 1: `metrics.csv` plus
`baseline_comparison.png` exist and actuated beats fixed, which is exactly what that
week's DoD asked for. Week 2: `week2_benchmark_metrics.csv` holds 3 seeds per
controller — fixed-time 16.39 s mean wait against PPO 4.00 s, a 75.6% reduction — so
"PPO beats fixed-time on the benchmark, 3-seed average" is met on the numbers.

### The Phase A–B failures, in one line each

- **Week 3** — single-intersection RL beats fixed-time on all three metrics but
  beats actuated on only one of three. Reported as failed rather than re-scoped.
- **Week 5** — the shared policy beats actuated on wait time (17.9 s vs 40.9 s) but
  **not reliably on throughput**: two seeds of three beat it, the third converged to
  a policy that jams the corridor and drags the mean below baseline. The fairness
  constraint measurably changed nothing. Week 4's simpler unshaped configuration
  does clear the bar — it is the Week 5 *shaping* that fails to add to it.

Both are genuine scientific results and stay in the final report. They are not
outstanding work items. A third failure — Week 9's federated DoD — is recorded
under Phase C below and shares a root cause with Week 5's.

## Phase C — feature sprint (Weeks 7–9)

| Week | Definition of Done | Verdict | Evidence |
|---|---|---|---|
| 7 | A natural-language query returns an answer grounded in real graph data | **MET** | `week7_report.md`, `week7_qa.json` |
| 8 | Vision detects vehicles in a held-out clip; injected anomaly flagged; one closure scenario end-to-end | **MET** (all three) | `week8_report.md` |
| 9 | Federated averaging improves a held-out district | **NOT MET** | `week9_report.md` |
| 9 | Fine-tuned model beats the base prompt on domain questions | **MET** | `week9_report.md` |

Week 7 runs against live backends: Neo4j AuraDB over Cypher, Chroma locally, and
`claude-opus-5` for phrasing. Verify with `python src\check_neo4j.py` and
`python src\week7_demo.py`.

### Week 8, in one line each

- **Vision** — YOLOv8n on 480 frames rendered from the twin, labelled from TraCI
  ground truth. mAP50 0.988 on junction C2, held out of training entirely.
- **Anomaly detection** — recall 1.00 on injected lane obstructions across 3
  seeds, 30 s mean detection latency, per-lane causal z-score detectors.
- **Scenario planner** — 18 closure × weather × seed runs complete. Weather
  dominates the closure: fog cuts throughput to 564 against 1598 in the clear.

### Week 9's third failure, and why it matters

The federated DoD is **NOT MET**, and the reason is the same argmax saturation
Weeks 5 and 6 documented. Separately trained district policies reach *byte-identical*
wait times on the held-out junction, so averaging them changes nothing — the
improvement over the local-only baseline is exactly 0.0%. This is the third
independent sighting of the effect, and it is a property of the action space
(binary next-phase, 5 s minimum green), not of the federation. The machinery
itself runs end to end and is unit-tested.

Priority routing and the emissions reward term are Week 9 deliverables without a
numeric DoD. Both are implemented and measured: emergency-vehicle wait falls
86% (416 s → 58 s) with no cost to general traffic.

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
