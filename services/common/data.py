"""Read-only access to the project's committed results, shared by the services.

Every figure a service returns is read from the same artifacts the reports and
the dossier are built from — the metrics CSVs, the graph document, the detector
metrics — so an API response and a report cannot disagree about a number.

Nothing here writes. The services are a read surface over work that already
happened; the training and evaluation harnesses remain the only things that
produce results.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import os
from typing import Any

from . import settings


def _read_csv(path: str) -> list[dict[str, str]]:
    """Read a CSV into a list of row dictionaries.

    Args:
        path: absolute path.

    Returns:
        The rows, or an empty list if the file is absent.
    """
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str) -> Any:
    """Read a JSON file.

    Args:
        path: absolute path.

    Returns:
        The parsed document, or None if the file is absent.
    """
    if not os.path.isfile(path):
        return None
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def outputs(*parts: str) -> str:
    """Build a path inside the mounted outputs directory.

    Args:
        *parts: path components.

    Returns:
        The joined absolute path.
    """
    return os.path.join(settings.OUTPUTS_DIR, *parts)


@functools.lru_cache(maxsize=1)
def corridor_runs() -> list[dict[str, Any]]:
    """Return every corridor-scope benchmark row.

    Returns:
        Rows from ``marl_metrics.csv`` restricted to corridor scope, with numeric
        fields converted.
    """
    rows = _read_csv(outputs("marl_metrics.csv"))
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("scope") != "corridor":
            continue
        record: dict[str, Any] = {
            "controller": row.get("controller"),
            "scenario": row.get("scenario"),
            "seed": int(row["seed"]) if row.get("seed") else None,
        }
        for key in ("avg_wait_time_s", "max_queue_len", "throughput_veh",
                    "total_co2_kg", "wait_p95_s", "worst_vehicle_wait_s"):
            value = row.get(key)
            record[key] = float(value) if value not in (None, "") else None
        out.append(record)
    return out


def summarise(scenario: str = "base") -> list[dict[str, Any]]:
    """Aggregate corridor runs into per-controller means for one scenario.

    Args:
        scenario: demand scenario to summarise.

    Returns:
        One record per controller, with the seed count behind each mean.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in corridor_runs():
        if row["scenario"] != scenario:
            continue
        grouped.setdefault(row["controller"], []).append(row)

    summary: list[dict[str, Any]] = []
    for controller, rows in sorted(grouped.items()):
        record: dict[str, Any] = {"controller": controller, "seeds": len(rows),
                                  "scenario": scenario}
        for key in ("avg_wait_time_s", "max_queue_len", "throughput_veh",
                    "total_co2_kg"):
            values = [r[key] for r in rows if r.get(key) is not None]
            record[key] = round(sum(values) / len(values), 3) if values else None
        summary.append(record)
    return summary


@functools.lru_cache(maxsize=1)
def scenarios() -> list[str]:
    """List the demand scenarios present in the metrics.

    Returns:
        Sorted scenario names.
    """
    return sorted({r["scenario"] for r in corridor_runs() if r.get("scenario")})


@functools.lru_cache(maxsize=1)
def graph_document() -> dict[str, Any] | None:
    """Load the corridor knowledge-graph document.

    Returns:
        The graph document, or None if it has not been built.
    """
    return _read_json(outputs("kg", "corridor_graph.json"))


@functools.lru_cache(maxsize=1)
def detector_metrics() -> dict[str, Any] | None:
    """Load the Week 8 detector's held-out metrics.

    Returns:
        The detector results, or None if the model has not been trained.
    """
    return _read_json(outputs("vision", "detector_results.json"))


@functools.lru_cache(maxsize=1)
def anomaly_results() -> dict[str, Any] | None:
    """Load the Week 8 anomaly-detection results.

    Returns:
        The results document, or None.
    """
    return _read_json(outputs("week8_anomaly_results.json"))


@functools.lru_cache(maxsize=1)
def federated_results() -> dict[str, Any] | None:
    """Load the Week 9 federated-learning results.

    Returns:
        The results document, or None.
    """
    return _read_json(outputs("week9_federated_results.json"))


@functools.lru_cache(maxsize=1)
def priority_results() -> dict[str, Any] | None:
    """Load the Week 9 priority-routing results.

    Returns:
        The results document, or None.
    """
    return _read_json(outputs("week9_priority_results.json"))


@functools.lru_cache(maxsize=1)
def scenario_planner_rows() -> list[dict[str, Any]]:
    """Load the Week 8 scenario-planner grid.

    Returns:
        Rows with numeric fields converted.
    """
    rows = _read_csv(outputs("week8_scenario_metrics.csv"))
    out: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"closure": row.get("closure"),
                                  "weather": row.get("weather"),
                                  "seed": int(row["seed"]) if row.get("seed") else None}
        for key in ("avg_wait_time_s", "max_queue_len", "throughput_veh",
                    "total_co2_kg"):
            value = row.get(key)
            record[key] = float(value) if value not in (None, "") else None
        out.append(record)
    return out


def data_available() -> dict[str, bool]:
    """Report which artifacts the mount actually contains.

    A service that cannot find its data should say so on its health endpoint
    rather than return empty lists that look like real answers.

    Returns:
        A map of artifact name to presence.
    """
    return {
        "corridor_metrics": bool(corridor_runs()),
        "knowledge_graph": graph_document() is not None,
        "detector_metrics": detector_metrics() is not None,
        "anomaly_results": anomaly_results() is not None,
        "federated_results": federated_results() is not None,
        "priority_results": priority_results() is not None,
        "scenario_planner": bool(scenario_planner_rows()),
    }
