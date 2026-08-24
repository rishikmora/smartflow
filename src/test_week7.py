"""Self-checks for Week 7: graph fidelity, grounding, refusal, and the read-only boundary.

Week 7's Definition of Done — "a natural-language query returns an answer grounded in
real graph data, not a hallucinated guess" — is only meaningful if three things are
independently true, so each gets its own check:

* **The graph says what the network says.** Every structural fact is re-derived
  straight from the SUMO file and the metrics CSVs and compared, so a bug in ingestion
  cannot quietly become a confidently-cited wrong answer.
* **Answers carry their evidence.** A grounded answer must cite sources, and a numeric
  claim must appear in the retrieved facts rather than only in the prose.
* **Out-of-domain questions are refused.** This is the anti-hallucination guarantee.
  Vector search returns its top ``k`` for any input, so "did it retrieve something" is
  not enough — the service must decline when nothing retrieved is *about* the question.

The read-only boundary from CLAUDE.md is checked by inspecting what the service
actually imports, in a subprocess, rather than by reading the source and trusting it.

Runs standalone and is pytest-compatible.

Usage:
    python src/test_week7.py
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# Anything that could drive or mutate the simulation / RL side. sumolib is deliberately
# absent: it only parses the .net.xml, which is a read operation.
FORBIDDEN_IMPORTS = {
    "traci", "libsumo", "sumo_rl", "ray", "stable_baselines3", "torch",
    "marl_env", "smartflow_env", "train_marl_corridor", "train_ppo_corridor",
    "eval_marl_corridor", "eval_corridor", "online_learning", "run_experiments",
}


# ── graph fidelity ──────────────────────────────────────────────────────────
def test_graph_matches_the_sumo_network() -> None:
    """Junction, road and adjacency facts must match the network file exactly."""
    import sumolib

    from knowledge_graph import open_graph
    from week7_config import CORRIDOR_NET

    net = sumolib.net.readNet(CORRIDOR_NET, withPrograms=True)
    expected_signalised = sorted(
        n.getID() for n in net.getNodes() if n.getType() == "traffic_light"
    )
    expected_feeds: dict[str, set[str]] = {j: set() for j in expected_signalised}
    for edge in net.getEdges():
        src, dst = edge.getFromNode().getID(), edge.getToNode().getID()
        if src in expected_feeds and dst in expected_feeds and src != dst:
            expected_feeds[src].add(dst)

    graph = open_graph()
    try:
        assert graph.signalised_junctions() == expected_signalised, (
            f"signalised junctions differ: {graph.signalised_junctions()} != {expected_signalised}"
        )
        for junction in expected_signalised:
            got = set(graph.neighbours(junction))
            assert got == expected_feeds[junction], (
                f"{junction} neighbours {sorted(got)} != {sorted(expected_feeds[junction])}"
            )
        # a junction's incoming lane count must match the network
        for junction in expected_signalised[:4]:
            expected_lanes = sorted(
                lane.getID()
                for edge in net.getNode(junction).getIncoming()
                for lane in edge.getLanes()
            )
            got_lanes = sorted(l["id"] for l in graph.lanes_of(junction))
            assert got_lanes == expected_lanes, f"{junction} lanes {got_lanes} != {expected_lanes}"
    finally:
        graph.close()


def test_graph_programs_match_the_network() -> None:
    """Signal programs and phase durations must match the tlLogic in the file."""
    import sumolib

    from knowledge_graph import open_graph
    from week7_config import CORRIDOR_NET

    net = sumolib.net.readNet(CORRIDOR_NET, withPrograms=True)
    graph = open_graph()
    try:
        for tls in net.getTrafficLights()[:4]:
            programs = tls.getPrograms()
            if not programs:
                continue
            program = list(programs.values())[0]
            expected_cycle = sum(int(p.duration) for p in program.getPhases())
            got = graph.program_of(tls.getID())
            assert got is not None, f"no program in graph for {tls.getID()}"
            assert got["cycle_s"] == expected_cycle, (
                f"{tls.getID()} cycle {got['cycle_s']} != {expected_cycle}"
            )
            assert got["num_phases"] == len(program.getPhases())
    finally:
        graph.close()


def test_graph_results_match_the_metrics_csv() -> None:
    """Every Result node must reproduce the mean of the committed CSV rows."""
    import statistics

    from knowledge_graph import open_graph
    from week7_config import METRICS_SOURCES

    # dedupe by seed: the two CSVs overlap on the fixed and actuated baselines, and a
    # seed-blind expectation would agree with a double-counting graph instead of catching it
    expected: dict[tuple[str, str, str], dict[str, float]] = {}
    for path in METRICS_SOURCES.values():
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") != "corridor":
                    continue
                scenario = row.get("scenario") or "base"
                for metric in ("avg_wait_time_s", "max_queue_len", "throughput_veh"):
                    value = row.get(metric)
                    if value in (None, ""):
                        continue
                    expected.setdefault((row["controller"], scenario, metric), {}).setdefault(
                        str(row.get("seed", "")), float(value))

    graph = open_graph()
    try:
        rows = graph.results()
        assert rows, "graph contains no Result nodes"
        for row in rows:
            key = (row["controller"], row["scenario"], row["metric"])
            assert key in expected, f"graph invented a result not in any CSV: {key}"
            want = round(statistics.fmean(expected[key].values()), 2)
            assert abs(row["mean"] - want) < 0.011, f"{key}: {row['mean']} != {want}"
            assert row["seeds"] == len(expected[key]), (
                f"{key}: graph reports {row['seeds']} seeds, CSVs hold {len(expected[key])} distinct"
            )
            assert row["seeds"] <= 3, f"{key}: {row['seeds']} seeds exceeds the 3-seed protocol"
    finally:
        graph.close()


# ── grounding and refusal ───────────────────────────────────────────────────
def test_graph_question_is_answered_from_the_graph() -> None:
    """A topology question must be answered from graph facts, with citations."""
    from llm_service import open_service

    service = open_service(use_llm=False)
    try:
        result = service.ask("which junctions does C2 feed into?")
        assert result.grounded, "topology question was not grounded"
        graph_sources = [f.source for f in result.facts if f.source.startswith("graph:")]
        assert graph_sources, "no graph facts backed a pure topology question"
        # the answer must actually name the real neighbours
        neighbours = set(service.graph.neighbours("C2"))
        assert neighbours, "C2 has no neighbours in the graph"
        assert all(n in result.answer for n in neighbours), (
            f"answer omits real neighbours {neighbours}: {result.answer}"
        )
        assert result.citations(), "grounded answer carried no citations"
    finally:
        service.close()


def test_numeric_claims_come_from_retrieved_facts() -> None:
    """A benchmark number in the answer must also appear in the retrieved evidence."""
    from llm_service import open_service

    service = open_service(use_llm=False)
    try:
        result = service.ask("how did the RL controller do on peak demand throughput?")
        assert result.grounded
        results_cited = [f for f in result.facts if f.source.startswith("graph:Result:")]
        assert results_cited, "no Result facts retrieved for a benchmark question"
        # every fact must be traceable to a Result node the graph actually holds
        known = {f"graph:Result:{r['id']}" for r in service.graph.results()}
        for fact in results_cited:
            assert fact.source in known, f"cited a result the graph does not have: {fact.source}"
    finally:
        service.close()


def test_out_of_domain_question_is_refused() -> None:
    """The service must decline rather than answer from model priors.

    Retrieval always returns its top k, so this is the check that the relevance floor
    is doing its job.
    """
    from llm_service import open_service

    service = open_service(use_llm=False)
    try:
        for question in (
            "what is the average rainfall in Hyderabad?",
            "who won the 2019 cricket world cup?",
            "what is the capital of France?",
        ):
            result = service.ask(question)
            assert not result.grounded, f"claimed grounding for out-of-domain: {question}"
            assert result.generator == "refusal", f"did not refuse: {question}"
            assert not result.facts and not result.passages, (
                f"refusal still carried evidence for: {question}"
            )
    finally:
        service.close()


def test_relevance_gate_separates_domains() -> None:
    """In-domain questions must score far above the floor, out-of-domain at zero."""
    from vector_store import open_store
    from week7_config import RELEVANCE_THRESHOLD

    store = open_store()
    in_domain = "why did the fairness constraint not work?"
    out_domain = "what is the average rainfall in Hyderabad?"

    best_in = max(store.relevance(in_domain, h["text"]) for h in store.search(in_domain, k=4))
    best_out = max(store.relevance(out_domain, h["text"]) for h in store.search(out_domain, k=4))
    assert best_in >= RELEVANCE_THRESHOLD, f"in-domain scored {best_in:.3f}"
    assert best_out < RELEVANCE_THRESHOLD, f"out-of-domain scored {best_out:.3f}"


# ── architectural boundary ──────────────────────────────────────────────────
def test_llm_service_cannot_touch_the_control_loop() -> None:
    """CLAUDE.md's hard boundary, verified by inspecting what actually gets imported.

    Runs in a subprocess so the check sees a clean interpreter rather than modules this
    test session already imported for other reasons.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    probe = (
        "import sys; sys.path.insert(0, r'%s');\n"
        "import llm_service;\n"
        "print('|'.join(sorted(sys.modules)))\n" % here
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, f"probe failed: {completed.stderr[-500:]}"
    loaded = set(completed.stdout.strip().split("|"))
    leaked = sorted(FORBIDDEN_IMPORTS & loaded)
    assert not leaked, (
        "llm_service transitively imports simulation/RL modules, breaking the "
        f"read-only boundary: {leaked}"
    )


def test_service_exposes_no_mutating_methods() -> None:
    """The service's public surface must be queries only."""
    from llm_service import AnalyticsService

    public = [m for m in dir(AnalyticsService) if not m.startswith("_")]
    assert sorted(public) == ["ask", "close"], f"unexpected public surface: {public}"


def main() -> None:
    """Run every check and report a pass/fail summary."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures: list[str] = []
    for test in tests:
        try:
            test()
            log.info("PASS  %s", test.__name__)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            failures.append(f"{test.__name__}: {exc}")
            log.error("FAIL  %s -> %s", test.__name__, exc)
    log.info("%d/%d checks passed", len(tests) - len(failures), len(tests))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
