# Week 10 — Platform Split (Phase D)

> **Status: SCAFFOLD ONLY — Phase D has not been executed.**
> This file sits alongside `week3_report.md`, `week4_report.md` and `week5_report.md`,
> which report measured results. This one does not. It records what exists on disk so
> that nobody mistakes the directory layout for a working platform.

## Definition of Done

> A push to main rebuilds and redeploys every service automatically, behind an
> authenticated dashboard.

**Verdict: NOT MET.** Nothing has been deployed, no dashboard exists, and no
authentication has been implemented. See the gap list below.

## What actually exists

Five FastAPI service directories, each a single `main.py` of 13–32 lines:

| Service | Endpoints | What they return |
|---|---|---|
| `services/sim_service` | `/health`, `/runs` | `{"status":"ok"}` and an empty list |
| `services/rl_service` | `/health` | `{"status":"ok"}` |
| `services/vision_service` | `/health` | `{"status":"ok"}` |
| `services/graph_service` | `/health`, `/network` | `{"status":"ok"}` and `{"nodes":[],"edges":[]}` |
| `services/llm_service` | `/health` | `{"status":"ok"}` |

**Every non-health endpoint is a hardcoded placeholder.** None of them reads the
metrics CSVs, loads a policy, touches SUMO, or calls the Week 7 knowledge graph.
`services/llm_service` shares a name with `src/llm_service.py` but contains none of
its logic — the working question-answering service is the one in `src/`.

Also present: `docker-compose.yml` and `docker-compose.prod.yml` (five services plus
Chroma and Redpanda), per-service Dockerfiles, `tests/test_health.py`, and a 26-line
`.github/workflows/ci.yml`.

## What has never been run

- No container has been built or started; the compose files are unexecuted.
- The CI workflow has never run — it requires a push to GitHub.
- `tests/test_health.py` has not been executed locally (`pytest` is not in the venv).
- No Auth0 tenant, no login, no authenticated dashboard.
- No Redpanda topic has been created and no message has been published.

## Why this is not a failure

Phase D is scheduled for Weeks 10–11 and the project is currently at the end of
Week 7. Nothing here is late. The scaffold was generated early by
`0ec5d2c "feat: scaffold smartflow remaining roadmap"` so that the eventual work
has a shape to fill in.

The reason this file was rewritten is narrower: as originally generated it opened
with "Implemented:" and a bullet list, which read like a completed week to anyone
skimming `outputs/`. The code it described was accurate; the framing was not.
