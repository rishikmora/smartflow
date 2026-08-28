"""LLM service: read-only analytics over the corridor.

Answers natural-language questions by calling the **graph service** over HTTP
and retrieving from the project's own committed reports. It demonstrates the
inter-service path: a question here becomes a graph query there.

**The hard boundary from CLAUDE.md is enforced structurally, not by convention.**
This service imports nothing from the simulation or RL side - no traci, no
sumo-rl, no stable_baselines3, no ray - and exposes no route that mutates
anything. ``tests/test_services.py`` asserts both: that the module graph stays
clean, and that the service's declared routes are read-only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from common import data, events, settings
from common.auth import auth_mode, require_user
from common.observability import instrument

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SERVICE = "llm_service"
GRAPH_SERVICE_URL = os.environ.get("GRAPH_SERVICE_URL", "http://graph_service:8000")
REPORT_DIR = settings.OUTPUTS_DIR
MAX_PASSAGES = 4

app = FastAPI(
    title="SmartFlow LLM Service",
    description="Read-only question answering grounded in the corridor graph "
                "and the project's own reports.",
    version="1.0.0",
)
instrument(app, SERVICE)



class QueryRequest(BaseModel):
    """A natural-language question."""

    question: str = Field(..., min_length=3, max_length=500)


def _call_graph(path: str) -> dict[str, Any] | None:
    """Call the graph service.

    Args:
        path: the path to request, e.g. ``"/graph/stats"``.

    Returns:
        The parsed response, or None if the service is unreachable.
    """
    url = GRAPH_SERVICE_URL.rstrip("/") + path
    request = urllib.request.Request(url)
    try:
        from common.auth import issue_dev_token
        if not settings.AUTH_DISABLED and not settings.auth0_configured():
            request.add_header("Authorization",
                               "Bearer " + issue_dev_token("llm_service"))
    except Exception:  # noqa: BLE001 - token minting is best effort
        pass
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("Graph service unreachable at %s: %s", url, exc)
        return None


def _retrieve(question: str) -> list[dict[str, str]]:
    """Retrieve report passages relevant to a question.

    A deliberately simple lexical match. The full Week 7 retrieval stack - Chroma
    plus the IDF relevance gate - lives in ``src/llm_service.py`` and needs the
    whole Python environment; this service is the thin HTTP surface, and its job
    is to show the service path rather than re-implement retrieval.

    Args:
        question: the user's question.

    Returns:
        Up to :data:`MAX_PASSAGES` passages with their source file.
    """
    terms = {t for t in re.findall(r"[a-z0-9_]{4,}", question.lower())}
    if not terms or not os.path.isdir(REPORT_DIR):
        return []

    scored: list[tuple[int, dict[str, str]]] = []
    for name in sorted(os.listdir(REPORT_DIR)):
        if not name.endswith(".md"):
            continue
        try:
            with open(os.path.join(REPORT_DIR, name), encoding="utf-8",
                      errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        for block in text.split("\n\n"):
            lowered = block.lower()
            hits = sum(1 for term in terms if term in lowered)
            if hits >= 2 and len(block) > 80:
                scored.append((hits, {"source": name, "text": block.strip()[:600]}))
    scored.sort(key=lambda pair: -pair[0])
    return [passage for _score, passage in scored[:MAX_PASSAGES]]


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Report service health and whether the graph service is reachable.

    Returns:
        Status and dependency detail.
    """
    graph = _call_graph("/health")
    return {
        "status": "ok",
        "service": SERVICE,
        "auth": auth_mode(),
        "events": events.broker_status(),
        "graph_service": {"url": GRAPH_SERVICE_URL,
                          "reachable": graph is not None,
                          "nodes": (graph or {}).get("nodes")},
        "read_only": True,
    }


@app.post("/query", tags=["analytics"])
def query(request: QueryRequest,
          _user: dict = Depends(require_user)) -> dict[str, Any]:
    """Answer a question from the graph and the project's reports.

    Args:
        request: the question.
        _user: authenticated principal.

    Returns:
        An answer with the evidence behind it, or an explicit refusal when
        nothing relevant was found.
    """
    question = request.question.strip()
    facts: list[dict[str, Any]] = []

    for junction_id in dict.fromkeys(re.findall(r"\b([A-D][0-3])\b", question.upper())):
        detail = _call_graph("/graph/junctions/" + junction_id)
        if detail:
            facts.append({
                "source": "graph:Junction:" + junction_id,
                "feeds": detail.get("feeds"),
                "lanes": len(detail.get("lanes") or []),
                "sensors": len(detail.get("sensors") or []),
            })

    passages = _retrieve(question)

    if not facts and not passages:
        events.publish(SERVICE, "query.refused", {"question": question[:120]})
        return {
            "question": question,
            "answer": "I do not have data for that. This service answers only "
                      "from the corridor knowledge graph and this project's own "
                      "reports.",
            "grounded": False,
            "facts": [],
            "passages": [],
        }

    parts: list[str] = []
    for fact in facts:
        junction = fact["source"].split(":")[-1]
        feeds = ", ".join(fact.get("feeds") or []) or "no downstream junctions"
        parts.append(f"{junction} feeds into {feeds}; it has {fact['lanes']} "
                     f"incoming lanes and {fact['sensors']} sensors.")
    if passages:
        parts.append("The project's reports add: " + passages[0]["text"][:300])

    events.publish(SERVICE, "query.answered",
                   {"question": question[:120], "facts": len(facts),
                    "passages": len(passages)})
    return {
        "question": question,
        "answer": " ".join(parts),
        "grounded": True,
        "facts": facts,
        "passages": passages,
    }


@app.get("/summary", tags=["analytics"])
def summary(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """Return the project's headline benchmark result.

    Args:
        _user: authenticated principal.

    Returns:
        The base-demand comparison, straight from the metrics.

    Raises:
        HTTPException: if the metrics are not mounted.
    """
    rows = data.summarise("base")
    if not rows:
        raise HTTPException(status_code=503, detail="Metrics not mounted.")
    return {"scenario": "base", "controllers": rows,
            "note": "3-seed means, read from outputs/marl_metrics.csv."}


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush buffered analytics events before the process exits."""
    events.close()
