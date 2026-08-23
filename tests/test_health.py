"""Health endpoint tests for SmartFlow services."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def _load_app(service: str):
    """Load a service FastAPI app from its main.py file."""
    path = Path(__file__).resolve().parents[1] / "services" / service / "main.py"
    spec = importlib.util.spec_from_file_location(f"{service}_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def test_health_endpoints() -> None:
    """Every service should expose GET /health."""
    for service in ["sim_service", "rl_service", "vision_service", "graph_service", "llm_service"]:
        client = TestClient(_load_app(service))
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
