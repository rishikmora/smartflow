"""Graph service: the corridor knowledge graph over HTTP.

Serves the Week 7 graph document - junctions, roads, lanes, signal programs,
sensors, rules and every measured result - without requiring a Neo4j instance.
The committed ``outputs/kg/corridor_graph.json`` is the same document the
embedded and AuraDB backends both load, so answers here match answers there.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query

from common import data, events
from common.auth import auth_mode, require_user
from common.observability import instrument

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SERVICE = "graph_service"
app = FastAPI(
    title="SmartFlow Graph Service",
    description="The corridor knowledge graph. Read-only.",
    version="1.0.0",
)
instrument(app, SERVICE)



def _document() -> dict[str, Any]:
    """Return the graph document.

    Returns:
        The parsed graph document.

    Raises:
        HTTPException: if it has not been built.
    """
    document = data.graph_document()
    if document is None:
        raise HTTPException(
            status_code=503,
            detail="Graph document not mounted. Run src/knowledge_graph.py.",
        )
    return document


def _nodes(label: str) -> list[dict[str, Any]]:
    """Return every node carrying one label.

    Args:
        label: the node label.

    Returns:
        Matching nodes.
    """
    return [n for n in _document()["nodes"] if n.get("label") == label]


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Report service health and graph size.

    Returns:
        Status plus node and relationship counts when available.
    """
    document = data.graph_document()
    return {
        "status": "ok" if document else "degraded",
        "service": SERVICE,
        "auth": auth_mode(),
        "events": events.broker_status(),
        "nodes": len(document["nodes"]) if document else 0,
        "relationships": len(document["edges"]) if document else 0,
    }


@app.get("/graph/stats", tags=["graph"])
def stats(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """Summarise the graph by node label and relationship type.

    Args:
        _user: authenticated principal.

    Returns:
        Counts per label and per relationship type.
    """
    document = _document()
    labels: dict[str, int] = {}
    for node in document["nodes"]:
        labels[node.get("label", "?")] = labels.get(node.get("label", "?"), 0) + 1
    kinds: dict[str, int] = {}
    for edge in document["edges"]:
        kinds[edge.get("type", "?")] = kinds.get(edge.get("type", "?"), 0) + 1
    return {"nodes": len(document["nodes"]), "relationships": len(document["edges"]),
            "by_label": dict(sorted(labels.items())),
            "by_type": dict(sorted(kinds.items()))}


@app.get("/graph/junctions", tags=["graph"])
def junctions(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """List every junction.

    Args:
        _user: authenticated principal.

    Returns:
        Junction ids with their coordinates and signalisation.
    """
    rows = [{"id": n["id"], "x": n.get("x"), "y": n.get("y"),
             "signalised": n.get("signalised")} for n in _nodes("Junction")]
    return {"count": len(rows), "junctions": sorted(rows, key=lambda r: r["id"])}


@app.get("/graph/junctions/{junction_id}", tags=["graph"])
def junction(junction_id: str, _user: dict = Depends(require_user)) -> dict[str, Any]:
    """Return one junction with its neighbours, lanes, sensors and program.

    Args:
        junction_id: the junction to describe.
        _user: authenticated principal.

    Returns:
        The junction and everything attached to it.

    Raises:
        HTTPException: if the junction is not in the graph.
    """
    document = _document()
    match = next((n for n in _nodes("Junction") if n["id"] == junction_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No junction {junction_id!r}")

    feeds = sorted(e["to"] for e in document["edges"]
                   if e.get("type") == "FEEDS" and e.get("from") == junction_id)
    fed_by = sorted(e["from"] for e in document["edges"]
                    if e.get("type") == "FEEDS" and e.get("to") == junction_id)
    # Lanes hang off roads, not off junctions: Road -HAS_LANE-> Lane and
    # Road -ENDS_AT-> Junction. Walking only the junction's own outbound edges
    # finds sensors and the program but reports zero lanes, which is what an
    # earlier version of this endpoint did.
    incoming_roads = {e["from"] for e in document["edges"]
                      if e.get("type") == "ENDS_AT" and e.get("to") == junction_id}
    outgoing_roads = {e["to"] for e in document["edges"]
                      if e.get("type") == "STARTS_AT" and e.get("from") == junction_id}
    lanes = sorted(e["to"] for e in document["edges"]
                   if e.get("type") == "HAS_LANE" and e.get("from") in incoming_roads)

    direct = {e["to"] for e in document["edges"] if e.get("from") == junction_id}
    sensors = [n["id"] for n in _nodes("Sensor") if n["id"] in direct]
    program = next((n for n in _nodes("Program") if n["id"] in direct), None)
    if program is not None:
        # Phases are their own nodes, linked Program -HAS_PHASE-> Phase. Returning
        # the Program node alone reports zero phases, which is what the dashboard
        # showed before this.
        phase_ids = {e["to"] for e in document["edges"]
                     if e.get("type") == "HAS_PHASE" and e.get("from") == program["id"]}
        phases = [n for n in _nodes("Phase") if n["id"] in phase_ids]
        program = dict(program)
        program["phases"] = sorted(phases, key=lambda n: n.get("index", 0))
        program["cycle_s"] = sum(int(p.get("duration_s", 0)) for p in phases)

    events.publish(SERVICE, "junction.read", {"junction": junction_id})
    return {"junction": match, "feeds": feeds, "fed_by": fed_by,
            "incoming_roads": sorted(incoming_roads),
            "outgoing_roads": sorted(outgoing_roads),
            "lanes": lanes, "sensors": sorted(sensors), "program": program}


@app.get("/graph/rules", tags=["graph"])
def rules(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """List the signal-timing rules the corridor is governed by.

    Args:
        _user: authenticated principal.

    Returns:
        The rule nodes.
    """
    return {"rules": sorted(_nodes("Rule"), key=lambda n: n["id"])}


@app.get("/graph/results", tags=["graph"])
def results(
    controller: str | None = Query(None),
    scenario: str | None = Query(None),
    _user: dict = Depends(require_user),
) -> dict[str, Any]:
    """Return measured results held in the graph.

    Args:
        controller: optional controller filter.
        scenario: optional scenario filter.
        _user: authenticated principal.

    Returns:
        Matching result nodes.
    """
    rows = _nodes("Result")
    if controller:
        rows = [r for r in rows if r.get("controller") == controller]
    if scenario:
        rows = [r for r in rows if r.get("scenario") == scenario]
    return {"count": len(rows), "results": sorted(rows, key=lambda r: r["id"])}


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush buffered analytics events before the process exits."""
    events.close()
