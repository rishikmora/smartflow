"""Simulation service API for SmartFlow."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SmartFlow Simulation Service")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "sim_service"}


@app.get("/runs")
def runs() -> list[dict[str, str]]:
    """Return benchmark run history placeholder."""
    return []
