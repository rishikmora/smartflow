"""Graph service API for SmartFlow."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SmartFlow Graph Service")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "graph_service"}


@app.get("/network")
def network() -> dict[str, list[dict[str, str]]]:
    """Return a lightweight graph placeholder for the dashboard."""
    return {"nodes": [], "edges": []}
