"""Week 6 demand scenarios: light, peak and asymmetric stress tests.

Week 6 asks whether a policy trained on one demand pattern still helps when the
demand changes. All scenarios are derived **deterministically from the base route
file** rather than regenerated with ``randomTrips.py``, so the underlying set of
origin-destination routes is held fixed and only the *volume* or *directional
balance* changes. That makes the comparison a controlled one: any difference in
results is attributable to demand, not to a different random road-usage pattern.

Scenarios
---------
``base``
    ``data/corridor.rou.xml`` unchanged — 1800 vehicles over 1800 s, the demand every
    policy in Weeks 3-5 was trained on.
``light``
    Every second vehicle kept (~50% demand). Signals matter least here; a good result
    is that RL does not *hurt* relative to fixed-time.
``peak``
    Base demand plus a duplicate of every second vehicle (~150%), pushing the corridor
    towards saturation.
``asymmetric``
    All east-west vehicles kept, north-south vehicles thinned to 40%. A policy that
    simply learned the base demand's balanced phase split should degrade here.

Grid geometry: junction ids are ``<column letter><row digit>``, so an edge ``A1B1``
runs east-west and ``A1A2`` runs north-south. A vehicle is classified by whichever
axis most of its edges travel along.

Generated files are written once into ``data/`` and reused; delete them to regenerate.
"""

from __future__ import annotations

import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_config import CORRIDOR_ACTUATED_NET, CORRIDOR_NET, CORRIDOR_ROUTE, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SCENARIOS = {
    "base": "the demand every policy was trained on (1800 vehicles / 1800 s)",
    "light": "~50% of base demand",
    "peak": "~150% of base demand",
    "asymmetric": "full east-west demand, north-south thinned to 40%",
}

PEAK_DUPLICATE_EVERY = 2      # duplicate every Nth vehicle -> ~150% demand
LIGHT_KEEP_EVERY = 2          # keep every Nth vehicle      -> ~50% demand
ASYMMETRIC_NS_KEEP = 0.4      # fraction of north-south vehicles retained
DUPLICATE_DEPART_OFFSET = 0.5  # seconds, so duplicates do not depart simultaneously


def _edge_axis(edge_id: str) -> str | None:
    """Classify one edge as east-west or north-south.

    Args:
        edge_id: an edge id of the form ``<from junction><to junction>``, e.g. ``A1B1``.

    Returns:
        ``"ew"``, ``"ns"``, or ``None`` if the id does not match the grid convention.
    """
    if len(edge_id) != 4:
        return None
    from_col, from_row, to_col, to_row = edge_id[0], edge_id[1], edge_id[2], edge_id[3]
    if from_col != to_col and from_row == to_row:
        return "ew"
    if from_col == to_col and from_row != to_row:
        return "ns"
    return None


def _vehicle_axis(vehicle: ET.Element) -> str:
    """Return the dominant travel axis of a vehicle's route.

    Args:
        vehicle: a ``<vehicle>`` element containing a ``<route edges="...">`` child.

    Returns:
        ``"ew"`` or ``"ns"``; ties resolve to ``"ew"``.
    """
    route = vehicle.find("route")
    if route is None:
        return "ew"
    counts = {"ew": 0, "ns": 0}
    for edge in (route.get("edges") or "").split():
        axis = _edge_axis(edge)
        if axis:
            counts[axis] += 1
    return "ns" if counts["ns"] > counts["ew"] else "ew"


def _load_base_vehicles() -> tuple[ET.Element, list[ET.Element]]:
    """Parse the base route file.

    Returns:
        ``(root, vehicles)`` where ``vehicles`` preserves document order.

    Raises:
        FileNotFoundError: if the base route file is missing.
    """
    if not os.path.isfile(CORRIDOR_ROUTE):
        raise FileNotFoundError(f"Base route file not found: {CORRIDOR_ROUTE}")
    try:
        tree = ET.parse(CORRIDOR_ROUTE)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse {CORRIDOR_ROUTE}: {exc}") from exc
    root = tree.getroot()
    return root, list(root.findall("vehicle"))


def _write_route_file(vehicles: list[ET.Element], path: str) -> None:
    """Write vehicles to a SUMO route file, sorted and renumbered.

    SUMO expects vehicles in non-decreasing departure order, so the list is sorted
    and ids are reassigned to stay unique after duplication.

    Args:
        vehicles: ``<vehicle>`` elements to write.
        path: destination ``.rou.xml``.
    """
    ordered = sorted(vehicles, key=lambda v: float(v.get("depart", "0")))
    root = ET.Element("routes")
    for index, vehicle in enumerate(ordered):
        clone = ET.fromstring(ET.tostring(vehicle))
        clone.set("id", str(index))
        clone.set("depart", f"{float(clone.get('depart', '0')):.2f}")
        root.append(clone)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
    log.info("Wrote %s (%d vehicles)", path, len(ordered))


def _build_light(vehicles: list[ET.Element]) -> list[ET.Element]:
    """Keep every ``LIGHT_KEEP_EVERY``-th vehicle."""
    return [v for i, v in enumerate(vehicles) if i % LIGHT_KEEP_EVERY == 0]


def _build_peak(vehicles: list[ET.Element]) -> list[ET.Element]:
    """Keep all vehicles and duplicate every ``PEAK_DUPLICATE_EVERY``-th one."""
    out = list(vehicles)
    for index, vehicle in enumerate(vehicles):
        if index % PEAK_DUPLICATE_EVERY != 0:
            continue
        clone = ET.fromstring(ET.tostring(vehicle))
        clone.set("depart", f"{float(vehicle.get('depart', '0')) + DUPLICATE_DEPART_OFFSET:.2f}")
        out.append(clone)
    return out


def _build_asymmetric(vehicles: list[ET.Element]) -> list[ET.Element]:
    """Keep all east-west vehicles and thin north-south ones deterministically.

    Thinning uses a fractional accumulator rather than an integer "keep every Nth"
    period, so a keep fraction of 0.4 really retains 40% — an integer period can only
    express 1/2, 1/3, 1/4 and would silently round 0.4 up to 0.5.
    """
    out: list[ET.Element] = []
    ns_seen = 0
    for vehicle in vehicles:
        if _vehicle_axis(vehicle) == "ew":
            out.append(vehicle)
            continue
        # keep this vehicle iff the retained-count crosses the next integer
        if int((ns_seen + 1) * ASYMMETRIC_NS_KEEP) > int(ns_seen * ASYMMETRIC_NS_KEEP):
            out.append(vehicle)
        ns_seen += 1
    return out


BUILDERS = {
    "light": _build_light,
    "peak": _build_peak,
    "asymmetric": _build_asymmetric,
}


def scenario_route_file(scenario: str, force: bool = False) -> str:
    """Return the route file for a scenario, generating it if needed.

    Args:
        scenario: one of :data:`SCENARIOS`.
        force: regenerate even if the file already exists.

    Returns:
        Absolute path to the scenario's ``.rou.xml``.

    Raises:
        ValueError: if ``scenario`` is unknown.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {sorted(SCENARIOS)}.")
    if scenario == "base":
        return CORRIDOR_ROUTE

    path = os.path.join(DATA_DIR, f"corridor_{scenario}.rou.xml")
    if os.path.isfile(path) and not force:
        return path

    _root, vehicles = _load_base_vehicles()
    _write_route_file(BUILDERS[scenario](vehicles), path)
    return path


def resolve_scenario(scenario: str, actuated: bool = False) -> tuple[str, str]:
    """Return the ``(net_file, route_file)`` pair for a scenario.

    Args:
        scenario: one of :data:`SCENARIOS`.
        actuated: return the all-actuated network instead of the fixed-time one.

    Returns:
        Tuple of absolute paths.
    """
    net = CORRIDOR_ACTUATED_NET if actuated else CORRIDOR_NET
    return net, scenario_route_file(scenario)


def describe() -> dict[str, dict[str, Any]]:
    """Summarise every scenario's demand, generating files as needed.

    Returns:
        ``{scenario: {"description": str, "vehicles": int, "route_file": str}}``.
    """
    summary: dict[str, dict[str, Any]] = {}
    for name, description in SCENARIOS.items():
        path = scenario_route_file(name)
        count = len(ET.parse(path).getroot().findall("vehicle"))
        summary[name] = {"description": description, "vehicles": count, "route_file": path}
    return summary


def main() -> None:
    """Generate every scenario route file and print a summary."""
    for name, info in describe().items():
        log.info("%-11s %5d vehicles  %s", name, info["vehicles"], info["description"])


if __name__ == "__main__":
    main()
