"""Simulation service: the corridor's network and its benchmark runs.

Serves what the simulation side of the project has produced — the network's
topology and every recorded benchmark run — from the same committed artifacts
the reports are built from.

It does **not** start SUMO. Running a 1800-second episode takes minutes and
belongs in the training harness, not behind a synchronous HTTP request; the
service is a read surface over runs that already happened.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ElementTree
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query

from common import data, events, settings
from common.auth import auth_mode, require_user
from common.observability import instrument

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SERVICE = "sim_service"
app = FastAPI(
    title="SmartFlow Simulation Service",
    description="Corridor topology and recorded benchmark runs. Read-only.",
    version="1.0.0",
)
instrument(app, SERVICE)



@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Report service health and what data the mount actually contains.

    Returns:
        Status, auth mode, event-broker state and artifact availability.
    """
    available = data.data_available()
    return {
        "status": "ok" if available["corridor_metrics"] else "degraded",
        "service": SERVICE,
        "auth": auth_mode(),
        "events": events.broker_status(),
        "data_root": settings.DATA_ROOT,
        "data": available,
    }


@app.get("/scenarios", tags=["runs"])
def list_scenarios(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """List the demand scenarios present in the benchmark data.

    Args:
        _user: authenticated principal.

    Returns:
        The scenario names.
    """
    return {"scenarios": data.scenarios()}


@app.get("/runs", tags=["runs"])
def list_runs(
    scenario: str | None = Query(None, description="Filter by demand scenario"),
    controller: str | None = Query(None, description="Filter by controller"),
    _user: dict = Depends(require_user),
) -> dict[str, Any]:
    """List recorded corridor-scope benchmark runs.

    Args:
        scenario: optional scenario filter.
        controller: optional controller filter.
        _user: authenticated principal.

    Returns:
        The matching runs, one per seed.
    """
    rows = data.corridor_runs()
    if scenario:
        rows = [r for r in rows if r["scenario"] == scenario]
    if controller:
        rows = [r for r in rows if r["controller"] == controller]
    events.publish(SERVICE, "runs.listed",
                   {"scenario": scenario, "controller": controller, "count": len(rows)})
    return {"count": len(rows), "runs": rows}


@app.get("/runs/summary", tags=["runs"])
def summary(
    scenario: str = Query("base", description="Demand scenario"),
    _user: dict = Depends(require_user),
) -> dict[str, Any]:
    """Return per-controller means for one scenario.

    Args:
        scenario: demand scenario.
        _user: authenticated principal.

    Returns:
        One record per controller, with the seed count behind each mean.

    Raises:
        HTTPException: if the scenario has no recorded runs.
    """
    rows = data.summarise(scenario)
    if not rows:
        raise HTTPException(status_code=404,
                            detail=f"No runs recorded for scenario {scenario!r}. "
                                   f"Known: {data.scenarios()}")
    return {"scenario": scenario, "controllers": rows}


@app.get("/network", tags=["network"])
def network(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """Summarise the corridor network read from the committed SUMO file.

    Args:
        _user: authenticated principal.

    Returns:
        Junction, edge and signal counts plus the signalised junction ids.

    Raises:
        HTTPException: if the network file is not on the mount.
    """
    path = os.path.join(settings.DATA_DIR, "corridor.net.xml")
    if not os.path.isfile(path):
        raise HTTPException(status_code=503,
                            detail=f"Network file not mounted at {path}")
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise HTTPException(status_code=500,
                            detail=f"Could not parse the network: {exc}") from exc

    junctions = [j for j in root.findall("junction")
                 if j.get("type") != "internal"]
    signalised = sorted(j.get("id") for j in junctions
                        if j.get("type") == "traffic_light")
    edges = [e for e in root.findall("edge") if e.get("function") != "internal"]
    return {
        "junctions": len(junctions),
        "signalised_junctions": len(signalised),
        "signalised_ids": signalised,
        "edges": len(edges),
        "source": "data/corridor.net.xml",
    }


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush buffered analytics events before the process exits."""
    events.close()
