"""RL service API for SmartFlow."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SmartFlow RL Service")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "rl_service"}
