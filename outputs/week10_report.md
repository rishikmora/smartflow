# Week 10 Report

Implemented:

- Five FastAPI service directories:
  - `services/sim_service`
  - `services/rl_service`
  - `services/vision_service`
  - `services/graph_service`
  - `services/llm_service`
- Each service exposes `GET /health`.
- Dockerfiles and minimal per-service requirements are present.
- `docker-compose.yml` includes the five services, Chroma, and Redpanda.
- `docker-compose.prod.yml` adds restart policy and memory limits.
- GitHub Actions workflow builds Docker images and runs health tests.

Verification:

- Python compilation passed for all service files.
- `pytest` is not installed in the local venv, so `tests/test_health.py` was not executed locally.

Deferred:

- Auth0 live token verification requires an Auth0 tenant and secrets.
- Redpanda topic creation and message flow require Docker runtime verification.
- CI pass status requires pushing to GitHub.
