"""RL service: trained policies and the results they produced.

Serves the reinforcement-learning side of the project: which policies exist on
disk, what they scored, and how they compare against the classical baselines.

Like the simulation service it starts nothing. Training a corridor policy takes
tens of minutes; the service reports what training produced.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query

from common import data, events, settings
from common.auth import auth_mode, require_user
from common.observability import instrument

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SERVICE = "rl_service"
BASELINES = {"fixed", "actuated"}

app = FastAPI(
    title="SmartFlow RL Service",
    description="Trained policies and their measured results. Read-only.",
    version="1.0.0",
)
instrument(app, SERVICE)



@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Report service health and data availability.

    Returns:
        Status, auth mode, event-broker state and artifact availability.
    """
    available = data.data_available()
    return {
        "status": "ok" if available["corridor_metrics"] else "degraded",
        "service": SERVICE,
        "auth": auth_mode(),
        "events": events.broker_status(),
        "data": available,
    }


@app.get("/models", tags=["models"])
def models(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """List trained policy artifacts present on the mount.

    Args:
        _user: authenticated principal.

    Returns:
        Model names with their sizes in bytes.
    """
    root = os.path.join(settings.DATA_ROOT, "models")
    found: list[dict[str, Any]] = []
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if os.path.isfile(path) and entry.endswith(".zip"):
                found.append({"name": entry, "kind": "sb3", "bytes": os.path.getsize(path)})
            elif os.path.isdir(path) and entry.startswith("marl_"):
                found.append({"name": entry, "kind": "rllib"})
    return {"count": len(found), "models": found}


@app.get("/results", tags=["results"])
def results(
    scenario: str = Query("base", description="Demand scenario"),
    _user: dict = Depends(require_user),
) -> dict[str, Any]:
    """Return per-controller means for one scenario.

    Args:
        scenario: demand scenario.
        _user: authenticated principal.

    Returns:
        One record per controller.

    Raises:
        HTTPException: if the scenario is unknown.
    """
    rows = data.summarise(scenario)
    if not rows:
        raise HTTPException(status_code=404,
                            detail=f"Unknown scenario {scenario!r}. Known: {data.scenarios()}")
    events.publish(SERVICE, "results.read", {"scenario": scenario})
    return {"scenario": scenario, "controllers": rows}


@app.get("/compare", tags=["results"])
def compare(
    scenario: str = Query("base"),
    metric: str = Query("avg_wait_time_s"),
    _user: dict = Depends(require_user),
) -> dict[str, Any]:
    """Compare every learned controller against the classical baselines.

    Args:
        scenario: demand scenario.
        metric: metric to compare on.
        _user: authenticated principal.

    Returns:
        Baselines, learned controllers and the percentage change against each
        baseline, with the direction of "better" stated explicitly.

    Raises:
        HTTPException: if the scenario or metric is unknown.
    """
    rows = data.summarise(scenario)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Unknown scenario {scenario!r}")
    if not any(metric in r for r in rows):
        raise HTTPException(status_code=400, detail=f"Unknown metric {metric!r}")

    # Throughput is the one metric where more is better.
    higher_is_better = metric == "throughput_veh"
    baselines = {r["controller"]: r for r in rows if r["controller"] in BASELINES}
    learned = [r for r in rows if r["controller"] not in BASELINES]

    comparisons: list[dict[str, Any]] = []
    for row in learned:
        value = row.get(metric)
        entry: dict[str, Any] = {"controller": row["controller"], "value": value,
                                 "seeds": row["seeds"], "versus": {}}
        for name, baseline in baselines.items():
            reference = baseline.get(metric)
            if value is None or not reference:
                continue
            change = (value - reference) / reference * 100
            entry["versus"][name] = {
                "baseline_value": reference,
                "change_pct": round(change, 2),
                "better": (change > 0) if higher_is_better else (change < 0),
            }
        comparisons.append(entry)

    return {
        "scenario": scenario,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "baselines": {k: v.get(metric) for k, v in baselines.items()},
        "comparisons": comparisons,
    }


@app.get("/federated", tags=["results"])
def federated(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """Return the Week 9 federated-learning result.

    Args:
        _user: authenticated principal.

    Returns:
        The recorded summary, including the verdict.

    Raises:
        HTTPException: if the experiment has not been run.
    """
    payload = data.federated_results()
    if payload is None:
        raise HTTPException(status_code=404,
                            detail="Federated results not present. Run src/federated.py.")
    return {
        "districts": payload["districts"],
        "held_out": payload["held_out"],
        "summary": payload["summary"],
        "weight_divergence": payload["per_seed"][0].get("weight_divergence"),
        "action_agreement": payload["per_seed"][0].get("action_agreement"),
    }


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush buffered analytics events before the process exits."""
    events.close()
