"""Corridor knowledge graph: roads, sensors, signal rules and measured results.

Everything in this graph is derived from files already in the repository — the SUMO
network for geometry and signal programs, the committed metrics CSVs for results. None
of it is authored by hand, so a question answered from the graph is answered from the
same artefacts the benchmark chapters were written from.

Schema
------
Nodes::

    Junction    a signalised or unsignalised node        {id, x, y, signalised, degree}
    Road        a directed link between two junctions    {id, from, to, length_m, axis}
    Lane        a lane belonging to a road               {id, road, length_m, width_m}
    Program     a traffic-light program                  {id, junction, type, cycle_s}
    Phase       one phase of a program                   {id, index, duration_s, state, kind}
    Sensor      per-lane detection an actuated program relies on  {id, junction, lane}
    Rule        a governing constant                     {id, value, unit, description, source}
    Controller  a control strategy that was benchmarked  {id, family}
    Scenario    a demand scenario                        {id, vehicles}
    Result      a measured metric                        {id, metric, mean, std, seeds}

Relationships::

    (Road)-[:STARTS_AT]->(Junction)      (Road)-[:ENDS_AT]->(Junction)
    (Road)-[:HAS_LANE]->(Lane)           (Junction)-[:FEEDS]->(Junction)
    (Junction)-[:RUNS]->(Program)        (Program)-[:HAS_PHASE]->(Phase)
    (Junction)-[:MONITORED_BY]->(Sensor) (Junction)-[:GOVERNED_BY]->(Rule)
    (Controller)-[:ACHIEVED]->(Result)   (Result)-[:ON]->(Scenario)

A note on Sensor nodes: the corridor network declares no detectors. SUMO's actuated
controller instantiates one induction loop per controlled lane implicitly, so these
nodes are *inferred* from the actuated program's controlled lanes and are labelled
``inferred: true``. They are included because "sensors" is part of what Week 7 asks the
graph to represent, and omitting them silently would misrepresent what the corridor
has; claiming they were declared in the file would be worse.

Backends
--------
:class:`Neo4jGraph` runs real Cypher against AuraDB. :class:`EmbeddedGraph` answers the
same typed queries from an in-process NetworkX graph. Callers use :func:`open_graph`
and never see the difference — which is what makes the AuraDB credential optional
rather than blocking.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week7_config import (
    CORRIDOR_ACTUATED_NET,
    CORRIDOR_NET,
    GRAPH_JSON,
    KG_DIR,
    METRICS_SOURCES,
    SIGNAL_RULES,
    neo4j_credentials,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def _use_system_trust_store() -> bool:
    """Make Python validate TLS against the OS certificate store.

    AuraDB's ``neo4j+s://`` scheme requires full certificate verification, and on some
    machines Python's bundled ``certifi`` chain rejects the Aura certificate with
    "self signed certificate in certificate chain" even though the operating system and
    every browser accept it — the root sits in the Windows store but not in certifi, or
    a TLS-inspecting proxy re-signs the connection. The failure surfaces confusingly, as
    ``ServiceUnavailable: Unable to retrieve routing information``.

    Delegating to the OS trust store fixes it without weakening verification, which is
    the important part: the alternative workarounds are ``neo4j+ssc://`` or disabling
    encryption, and both actually stop checking who they are talking to.

    Returns:
        Whether the trust store was successfully installed.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except ImportError:
        log.debug("truststore unavailable; using certifi's default chain.")
    except Exception as exc:  # pragma: no cover - platform dependent
        log.warning("Could not install the system trust store: %s", exc)
    return False


# ── ingestion ───────────────────────────────────────────────────────────────
def _axis_of(from_id: str, to_id: str) -> str:
    """Classify a link as east-west or north-south from the grid's node naming.

    Junction ids are ``<column letter><row digit>``, so a link whose column changes
    runs east-west and one whose row changes runs north-south.

    Args:
        from_id: upstream junction id.
        to_id: downstream junction id.

    Returns:
        ``"ew"``, ``"ns"`` or ``"other"``.
    """
    if len(from_id) < 2 or len(to_id) < 2:
        return "other"
    if from_id[0] != to_id[0] and from_id[1:] == to_id[1:]:
        return "ew"
    if from_id[0] == to_id[0] and from_id[1:] != to_id[1:]:
        return "ns"
    return "other"


def build_document(net_file: str = CORRIDOR_NET,
                   actuated_net: str = CORRIDOR_ACTUATED_NET) -> dict[str, Any]:
    """Extract the whole graph from the SUMO network and the metrics CSVs.

    Args:
        net_file: fixed-time network, the source for geometry and static programs.
        actuated_net: actuated network, the source for inferred sensors.

    Returns:
        ``{"nodes": [...], "edges": [...]}`` — a backend-independent document.

    Raises:
        FileNotFoundError: if the network file is missing.
    """
    if not os.path.isfile(net_file):
        raise FileNotFoundError(f"SUMO network not found: {net_file}")
    import sumolib

    # withPrograms is required: without it sumolib parses no tlLogic at all and
    # getPrograms() silently returns {}, producing a graph with no signal data.
    net = sumolib.net.readNet(net_file, withPrograms=True)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    signalised = {n.getID() for n in net.getNodes() if n.getType() == "traffic_light"}

    for node in net.getNodes():
        x, y = node.getCoord()
        nodes.append({
            "label": "Junction", "id": node.getID(),
            "x": round(x, 1), "y": round(y, 1),
            "signalised": node.getID() in signalised,
            "incoming": len(node.getIncoming()), "outgoing": len(node.getOutgoing()),
        })

    for edge in net.getEdges():
        src, dst = edge.getFromNode().getID(), edge.getToNode().getID()
        road_id = edge.getID()
        nodes.append({
            "label": "Road", "id": road_id, "from": src, "to": dst,
            "length_m": round(edge.getLength(), 1),
            "speed_limit_ms": round(edge.getSpeed(), 1),
            "axis": _axis_of(src, dst),
            "lanes": len(edge.getLanes()),
        })
        edges.append({"type": "STARTS_AT", "from": road_id, "to": src})
        edges.append({"type": "ENDS_AT", "from": road_id, "to": dst})
        if src in signalised and dst in signalised:
            edges.append({"type": "FEEDS", "from": src, "to": dst})
        for lane in edge.getLanes():
            nodes.append({
                "label": "Lane", "id": lane.getID(), "road": road_id,
                "length_m": round(lane.getLength(), 1),
                "width_m": round(lane.getWidth(), 2),
            })
            edges.append({"type": "HAS_LANE", "from": road_id, "to": lane.getID()})

    _add_programs(net, signalised, nodes, edges)
    _add_sensors(actuated_net, signalised, nodes, edges)
    _add_rules(signalised, nodes, edges)
    _add_results(nodes, edges)

    log.info("Built knowledge graph: %d nodes, %d relationships", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}


def _add_programs(net: Any, signalised: set[str], nodes: list, edges: list) -> None:
    """Add Program and Phase nodes from the network's traffic-light logic."""
    for tls in net.getTrafficLights():
        junction = tls.getID()
        if junction not in signalised:
            continue
        programs = tls.getPrograms()
        for program_id, program in programs.items():
            phases = program.getPhases()
            cycle = sum(int(p.duration) for p in phases)
            node_id = f"{junction}:{program_id}"
            nodes.append({
                "label": "Program", "id": node_id, "junction": junction,
                "program_id": program_id, "type": program.getType(),
                "cycle_s": cycle, "num_phases": len(phases),
            })
            edges.append({"type": "RUNS", "from": junction, "to": node_id})
            for index, phase in enumerate(phases):
                state = phase.state
                kind = "yellow" if "y" in state else ("green" if "G" in state or "g" in state else "red")
                phase_id = f"{node_id}:{index}"
                nodes.append({
                    "label": "Phase", "id": phase_id, "program": node_id,
                    "index": index, "duration_s": int(phase.duration),
                    "state": state, "kind": kind,
                })
                edges.append({"type": "HAS_PHASE", "from": node_id, "to": phase_id})


def _add_sensors(actuated_net: str, signalised: set[str], nodes: list, edges: list) -> None:
    """Add inferred per-lane Sensor nodes for junctions that run an actuated program.

    SUMO does not declare these in the network file — an actuated ``tlLogic`` gets one
    induction loop per controlled lane implicitly. They are marked ``inferred`` so an
    answer can say where they come from.
    """
    if not os.path.isfile(actuated_net):
        log.warning("Actuated network missing (%s); no Sensor nodes added.", actuated_net)
        return
    import sumolib

    net = sumolib.net.readNet(actuated_net, withPrograms=True)
    for tls in net.getTrafficLights():
        junction = tls.getID()
        if junction not in signalised:
            continue
        is_actuated = any(
            program.getType() == "actuated"
            for program in tls.getPrograms().values()
        )
        if not is_actuated:
            continue
        lanes = sorted({conn[0].getID() for conn in tls.getConnections()})
        for lane_id in lanes:
            sensor_id = f"det:{junction}:{lane_id}"
            nodes.append({
                "label": "Sensor", "id": sensor_id, "junction": junction,
                "lane": lane_id, "kind": "induction_loop", "inferred": True,
                "note": "instantiated implicitly by SUMO for actuated control; not declared in the .net.xml",
            })
            edges.append({"type": "MONITORED_BY", "from": junction, "to": sensor_id})


def _add_rules(signalised: set[str], nodes: list, edges: list) -> None:
    """Add Rule nodes for the signal-timing constants, and attach them to junctions."""
    for name, value, unit, description, source in SIGNAL_RULES:
        nodes.append({
            "label": "Rule", "id": name, "value": value, "unit": unit,
            "description": description, "source": source,
        })
        for junction in sorted(signalised):
            edges.append({"type": "GOVERNED_BY", "from": junction, "to": name})


def _add_results(nodes: list, edges: list) -> None:
    """Fold the committed benchmark CSVs into Controller / Scenario / Result nodes."""
    import statistics

    seen_controllers: set[str] = set()
    seen_scenarios: set[str] = set()
    # Keyed by seed as well as by measurement: the Week 3 and corridor CSVs both carry
    # `fixed` and `actuated` base rows for seeds 0-2, so a seed-blind key counted every
    # baseline twice and reported "6 seeds" for a 3-seed result.
    observations: dict[tuple[str, str, str], dict[str, float]] = {}

    for scope_name, path in METRICS_SOURCES.items():
        if not os.path.isfile(path):
            log.warning("Metrics file missing, skipping: %s", path)
            continue
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") != "corridor":
                    continue
                controller = row["controller"]
                scenario = row.get("scenario") or "base"
                for metric in ("avg_wait_time_s", "max_queue_len", "throughput_veh"):
                    raw = row.get(metric)
                    if raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    seed = str(row.get("seed", ""))
                    # first file wins for a given seed; a repeat is the same measurement
                    observations.setdefault((controller, scenario, metric), {}).setdefault(seed, value)
                seen_controllers.add(controller)
                seen_scenarios.add(scenario)

    for controller in sorted(seen_controllers):
        family = ("reinforcement learning" if controller.startswith("marl") or controller.startswith("ppo")
                  else "classical")
        nodes.append({"label": "Controller", "id": controller, "family": family})
    for scenario in sorted(seen_scenarios):
        nodes.append({"label": "Scenario", "id": scenario})

    for (controller, scenario, metric), by_seed in sorted(observations.items()):
        values = [by_seed[k] for k in sorted(by_seed)]
        result_id = f"{controller}|{scenario}|{metric}"
        nodes.append({
            "label": "Result", "id": result_id, "controller": controller,
            "scenario": scenario, "metric": metric,
            "mean": round(statistics.fmean(values), 2),
            "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 2),
            "seeds": len(values),
        })
        edges.append({"type": "ACHIEVED", "from": controller, "to": result_id})
        edges.append({"type": "ON", "from": result_id, "to": scenario})


# ── query interface (both backends implement this) ──────────────────────────
class GraphQueries:
    """The typed queries the LLM service is allowed to ask.

    Deliberately not a raw-Cypher passthrough. Every method returns plain dicts with a
    stable shape, so the answering layer composes facts it can cite rather than
    interpreting free-form query output — and so the embedded backend can implement
    exactly the same contract without a Cypher engine.
    """

    backend: str = "abstract"

    def stats(self) -> dict[str, Any]: raise NotImplementedError
    def junction(self, junction_id: str) -> dict[str, Any] | None: raise NotImplementedError
    def neighbours(self, junction_id: str) -> list[str]: raise NotImplementedError
    def lanes_of(self, junction_id: str) -> list[dict[str, Any]]: raise NotImplementedError
    def program_of(self, junction_id: str) -> dict[str, Any] | None: raise NotImplementedError
    def sensors_of(self, junction_id: str) -> list[dict[str, Any]]: raise NotImplementedError
    def rules(self) -> list[dict[str, Any]]: raise NotImplementedError
    def results(self, controller: str | None = None, scenario: str | None = None,
                metric: str | None = None) -> list[dict[str, Any]]: raise NotImplementedError
    def signalised_junctions(self) -> list[str]: raise NotImplementedError
    def roads_between(self, a: str, b: str) -> list[dict[str, Any]]: raise NotImplementedError
    def close(self) -> None: return None


class EmbeddedGraph(GraphQueries):
    """In-process backend over NetworkX, used when AuraDB is not configured."""

    backend = "embedded (networkx)"

    def __init__(self, document: dict[str, Any]) -> None:
        """Index the graph document for querying.

        Args:
            document: output of :func:`build_document`.
        """
        import networkx as nx

        self.doc = document
        self.by_id = {n["id"]: n for n in document["nodes"]}
        self.by_label: dict[str, list[dict]] = {}
        for node in document["nodes"]:
            self.by_label.setdefault(node["label"], []).append(node)
        self.g = nx.MultiDiGraph()
        for node in document["nodes"]:
            self.g.add_node(node["id"], **node)
        for edge in document["edges"]:
            self.g.add_edge(edge["from"], edge["to"], key=edge["type"], type=edge["type"])

    def _out(self, node_id: str, rel: str) -> list[str]:
        return [v for _u, v, k in self.g.out_edges(node_id, keys=True) if k == rel]

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "nodes": self.g.number_of_nodes(),
            "relationships": self.g.number_of_edges(),
            "labels": {label: len(items) for label, items in sorted(self.by_label.items())},
        }

    def junction(self, junction_id: str) -> dict[str, Any] | None:
        node = self.by_id.get(junction_id)
        return node if node and node.get("label") == "Junction" else None

    def neighbours(self, junction_id: str) -> list[str]:
        return sorted(self._out(junction_id, "FEEDS"))

    def lanes_of(self, junction_id: str) -> list[dict[str, Any]]:
        out = []
        for road in self.by_label.get("Road", []):
            if road["to"] != junction_id:
                continue
            for lane_id in self._out(road["id"], "HAS_LANE"):
                out.append(self.by_id[lane_id])
        return sorted(out, key=lambda item: item["id"])

    def program_of(self, junction_id: str) -> dict[str, Any] | None:
        for program_id in self._out(junction_id, "RUNS"):
            program = dict(self.by_id[program_id])
            program["phases"] = [self.by_id[p] for p in self._out(program_id, "HAS_PHASE")]
            return program
        return None

    def sensors_of(self, junction_id: str) -> list[dict[str, Any]]:
        return [self.by_id[s] for s in sorted(self._out(junction_id, "MONITORED_BY"))]

    def rules(self) -> list[dict[str, Any]]:
        return sorted(self.by_label.get("Rule", []), key=lambda item: item["id"])

    def results(self, controller: str | None = None, scenario: str | None = None,
                metric: str | None = None) -> list[dict[str, Any]]:
        out = []
        for node in self.by_label.get("Result", []):
            if controller and node["controller"] != controller:
                continue
            if scenario and node["scenario"] != scenario:
                continue
            if metric and node["metric"] != metric:
                continue
            out.append(node)
        return sorted(out, key=lambda item: (item["scenario"], item["metric"], item["controller"]))

    def signalised_junctions(self) -> list[str]:
        return sorted(n["id"] for n in self.by_label.get("Junction", []) if n.get("signalised"))

    def roads_between(self, a: str, b: str) -> list[dict[str, Any]]:
        return [r for r in self.by_label.get("Road", []) if r["from"] == a and r["to"] == b]


class Neo4jGraph(GraphQueries):
    """AuraDB backend. Same contract, implemented in Cypher."""

    backend = "neo4j aura"

    def __init__(self, credentials: dict[str, str]) -> None:
        """Open a driver session against AuraDB.

        Args:
            credentials: output of :func:`week7_config.neo4j_credentials`.
        """
        _use_system_trust_store()
        from neo4j import GraphDatabase

        self.database = credentials["database"]
        self.driver = GraphDatabase.driver(
            credentials["uri"], auth=(credentials["user"], credentials["password"])
        )
        self.driver.verify_connectivity()
        log.info("Connected to Neo4j at %s", credentials["uri"])

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(cypher, **params)]

    def load(self, document: dict[str, Any], wipe: bool = True) -> None:
        """Write the graph document into AuraDB.

        Args:
            document: output of :func:`build_document`.
            wipe: delete existing SmartFlow nodes first, so loads are idempotent.
        """
        if wipe:
            self._run("MATCH (n:SmartFlow) DETACH DELETE n")
        for node in document["nodes"]:
            props = {k: v for k, v in node.items() if k != "label"}
            self._run(
                f"MERGE (n:SmartFlow:{node['label']} {{id: $id}}) SET n += $props",
                id=node["id"], props=props,
            )
        for edge in document["edges"]:
            self._run(
                f"MATCH (a:SmartFlow {{id: $from}}), (b:SmartFlow {{id: $to}}) "
                f"MERGE (a)-[:{edge['type']}]->(b)",
                **{"from": edge["from"], "to": edge["to"]},
            )
        log.info("Loaded %d nodes and %d relationships into Neo4j",
                 len(document["nodes"]), len(document["edges"]))

    def stats(self) -> dict[str, Any]:
        counts = self._run(
            "MATCH (n:SmartFlow) RETURN labels(n) AS labels, count(*) AS c"
        )
        rels = self._run("MATCH (:SmartFlow)-[r]->(:SmartFlow) RETURN count(r) AS c")
        by_label: dict[str, int] = {}
        for row in counts:
            for label in row["labels"]:
                if label != "SmartFlow":
                    by_label[label] = by_label.get(label, 0) + row["c"]
        return {
            "backend": self.backend,
            "nodes": sum(by_label.values()),
            "relationships": rels[0]["c"] if rels else 0,
            "labels": dict(sorted(by_label.items())),
        }

    def junction(self, junction_id: str) -> dict[str, Any] | None:
        rows = self._run("MATCH (j:Junction {id: $id}) RETURN properties(j) AS p", id=junction_id)
        return rows[0]["p"] if rows else None

    def neighbours(self, junction_id: str) -> list[str]:
        rows = self._run(
            "MATCH (:Junction {id: $id})-[:FEEDS]->(n:Junction) RETURN n.id AS id ORDER BY n.id",
            id=junction_id,
        )
        return [r["id"] for r in rows]

    def lanes_of(self, junction_id: str) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (r:Road)-[:ENDS_AT]->(:Junction {id: $id}) "
            "MATCH (r)-[:HAS_LANE]->(l:Lane) RETURN properties(l) AS p ORDER BY l.id",
            id=junction_id,
        )
        return [r["p"] for r in rows]

    def program_of(self, junction_id: str) -> dict[str, Any] | None:
        rows = self._run(
            "MATCH (:Junction {id: $id})-[:RUNS]->(p:Program) "
            "OPTIONAL MATCH (p)-[:HAS_PHASE]->(ph:Phase) "
            "RETURN properties(p) AS prog, collect(properties(ph)) AS phases",
            id=junction_id,
        )
        if not rows:
            return None
        program = dict(rows[0]["prog"])
        program["phases"] = sorted(rows[0]["phases"], key=lambda item: item.get("index", 0))
        return program

    def sensors_of(self, junction_id: str) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (:Junction {id: $id})-[:MONITORED_BY]->(s:Sensor) "
            "RETURN properties(s) AS p ORDER BY s.id",
            id=junction_id,
        )
        return [r["p"] for r in rows]

    def rules(self) -> list[dict[str, Any]]:
        rows = self._run("MATCH (r:Rule) RETURN properties(r) AS p ORDER BY r.id")
        return [r["p"] for r in rows]

    def results(self, controller: str | None = None, scenario: str | None = None,
                metric: str | None = None) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (c:Controller)-[:ACHIEVED]->(r:Result)-[:ON]->(s:Scenario) "
            "WHERE ($controller IS NULL OR c.id = $controller) "
            "AND ($scenario IS NULL OR s.id = $scenario) "
            "AND ($metric IS NULL OR r.metric = $metric) "
            "RETURN properties(r) AS p ORDER BY s.id, r.metric, c.id",
            controller=controller, scenario=scenario, metric=metric,
        )
        return [r["p"] for r in rows]

    def signalised_junctions(self) -> list[str]:
        rows = self._run(
            "MATCH (j:Junction {signalised: true}) RETURN j.id AS id ORDER BY j.id"
        )
        return [r["id"] for r in rows]

    def roads_between(self, a: str, b: str) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH (r:Road)-[:STARTS_AT]->(:Junction {id: $a}) "
            "MATCH (r)-[:ENDS_AT]->(:Junction {id: $b}) RETURN properties(r) AS p",
            a=a, b=b,
        )
        return [r["p"] for r in rows]

    def close(self) -> None:
        self.driver.close()


def save_document(document: dict[str, Any], path: str = GRAPH_JSON) -> str:
    """Persist the graph document so the embedded backend can reload it cheaply."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
    log.info("Wrote %s", path)
    return path


def load_document(path: str = GRAPH_JSON) -> dict[str, Any]:
    """Load a persisted graph document, rebuilding it from SUMO if absent."""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    document = build_document()
    save_document(document, path)
    return document


def open_graph(prefer_neo4j: bool = True) -> GraphQueries:
    """Open the best available graph backend.

    Args:
        prefer_neo4j: try AuraDB first when credentials are present.

    Returns:
        A :class:`GraphQueries` implementation. Falls back to the embedded backend if
        AuraDB is unconfigured or unreachable, logging which was chosen.
    """
    document = load_document()
    if prefer_neo4j:
        credentials = neo4j_credentials()
        if credentials:
            try:
                graph = Neo4jGraph(credentials)
                if graph.stats()["nodes"] == 0:
                    graph.load(document)
                return graph
            except Exception as exc:
                log.warning("Neo4j unavailable (%s); using the embedded graph instead.", exc)
    return EmbeddedGraph(document)


def main() -> None:
    """Rebuild the graph from source files and report what it contains."""
    os.makedirs(KG_DIR, exist_ok=True)
    document = build_document()
    save_document(document)
    graph = open_graph()
    try:
        stats = graph.stats()
        log.info("backend=%s nodes=%d relationships=%d", stats["backend"], stats["nodes"], stats["relationships"])
        for label, count in stats["labels"].items():
            log.info("  %-11s %d", label, count)
    finally:
        graph.close()


if __name__ == "__main__":
    main()
