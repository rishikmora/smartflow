"""Week 7 demonstration: run a question set through the service and write the report.

The question set is chosen to exercise each retrieval path separately, so the
transcript shows *where* every answer came from rather than only that one came back:

* pure graph questions, answerable only from the network topology;
* rule questions, answerable only from the signal-timing constants;
* result questions, answerable only from the committed metrics;
* prose questions, answerable only from the written reports;
* out-of-domain questions, which must be **refused**.

That last group is the point. A RAG service that answers everything has not
demonstrated grounding — it has demonstrated fluency. Week 7's Definition of Done asks
for an answer "grounded in real graph data, not a hallucinated guess", so the demo has
to show the guess being declined.

Usage:
    python src/week7_demo.py
    python src/week7_demo.py --no-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_service import open_service
from week7_config import QA_JSON, WEEK7_REPORT, anthropic_available, neo4j_credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

QUESTIONS = [
    ("graph: topology", "Which junctions does C2 feed into?"),
    ("graph: topology", "How many signalised junctions does the corridor network have?"),
    ("graph: lanes", "How many incoming lanes does junction B1 have?"),
    ("graph: sensors", "What sensors does junction B1 have?"),
    ("graph: signal program", "What phases does junction B1 run and how long is its cycle?"),
    ("graph: rules", "What is the minimum green time and where is it defined?"),
    ("graph: results", "How did the RL controller do on peak demand throughput?"),
    ("graph: results", "What average wait did actuated control achieve on base demand?"),
    ("documents: why", "Why did the fairness constraint fail to change anything?"),
    ("documents: why", "Why did Week 3's Definition of Done fail?"),
    ("refusal", "What is the average rainfall in Hyderabad?"),
    ("refusal", "Who won the 2019 cricket world cup?"),
]


def run(use_llm: bool = True) -> list[dict]:
    """Ask every question and collect the grounded answers.

    Args:
        use_llm: allow Claude API calls; ``False`` forces the extractive answerer.

    Returns:
        Serialised answers, one per question, with their category attached.
    """
    service = open_service(use_llm=use_llm)
    records: list[dict] = []
    try:
        for category, question in QUESTIONS:
            result = service.ask(question)
            record = result.to_dict()
            record["category"] = category
            records.append(record)
            log.info("[%-18s] %-55s -> %s (%d facts, %d passages)",
                     category, question[:55], result.generator,
                     len(result.facts), len(result.passages))
    finally:
        service.close()

    os.makedirs(os.path.dirname(QA_JSON), exist_ok=True)
    with open(QA_JSON, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    log.info("Wrote %s", QA_JSON)
    return records


def write_report(records: list[dict]) -> None:
    """Write ``outputs/week7_report.md`` from the transcript.

    Args:
        records: output of :func:`run`.
    """
    from knowledge_graph import open_graph

    graph = open_graph()
    try:
        stats = graph.stats()
    finally:
        graph.close()

    grounded = [r for r in records if r["grounded"]]
    refusals = [r for r in records if not r["grounded"]]
    expected_refusals = [r for r in records if r["category"] == "refusal"]
    correct_refusals = [r for r in expected_refusals if not r["grounded"]]
    false_refusals = [r for r in refusals if r["category"] != "refusal"]
    backends = records[0]["backends"] if records else {}

    parts: list[str] = ["# Week 7 — Knowledge Graph, RAG and the Read-Only LLM Service\n\n"]
    parts.append(
        "A question-answering layer over the corridor. The graph holds the road network,\n"
        "its signal programs, the rules that govern them and every measured result; the\n"
        "vector store holds this project's own reports. The service queries **both** before\n"
        "answering, and returns the evidence alongside the answer so any sentence can be\n"
        "checked against a source.\n\n"
    )

    parts.append("## What the graph contains\n\n")
    parts.append(f"`{stats['backend']}` — **{stats['nodes']} nodes**, "
                 f"**{stats['relationships']} relationships**.\n\n")
    parts.append("| Node type | Count | Derived from |\n|---|---|---|\n")
    provenance = {
        "Junction": "`corridor.net.xml` nodes",
        "Road": "`corridor.net.xml` edges",
        "Lane": "`corridor.net.xml` lanes",
        "Program": "`tlLogic` blocks in the network",
        "Phase": "phases inside each `tlLogic`",
        "Sensor": "inferred: one induction loop per controlled lane of an actuated program",
        "Rule": "signal-timing constants in `week4_config.py`",
        "Controller": "controllers present in the metrics CSVs",
        "Scenario": "demand scenarios present in the metrics CSVs",
        "Result": "3-seed means from `marl_metrics.csv` and `week3_corridor_metrics.csv`",
    }
    for label, count in stats["labels"].items():
        parts.append(f"| `{label}` | {count} | {provenance.get(label, '—')} |\n")
    parts.append(
        "\nNothing here is hand-authored. `src/test_week7.py` re-derives the topology, the\n"
        "signal programs and every result mean straight from the source files and compares\n"
        "them, so an ingestion bug cannot become a confidently-cited wrong answer.\n"
    )
    parts.append(
        "\n**On Sensor nodes.** The network declares no detectors. SUMO instantiates one\n"
        "induction loop per controlled lane implicitly when a program is actuated, so these\n"
        "nodes are inferred and carry `inferred: true` plus a note saying so. They are\n"
        "included because 'sensors' is part of what this week asks the graph to model;\n"
        "presenting them as declared infrastructure would be the dishonest option.\n"
    )

    parts.append("\n## Definition of Done\n\n")
    parts.append("> A natural-language query returns an answer grounded in real graph data,\n"
                 "> not a hallucinated guess.\n\n")
    parts.append(f"- Questions asked: **{len(records)}**\n")
    parts.append(f"- Grounded answers: **{len(grounded)}**, each carrying its citations\n")
    parts.append(f"- Out-of-domain questions correctly refused: "
                 f"**{len(correct_refusals)} / {len(expected_refusals)}**\n")
    parts.append(f"- In-domain questions wrongly refused: **{len(false_refusals)}**\n")
    met = (len(correct_refusals) == len(expected_refusals)
           and not false_refusals
           and all(r["citations"] for r in grounded))
    parts.append(f"\n**Verdict: {'MET' if met else 'NOT MET'}**\n")

    parts.append(
        "\nThe refusals are the part that matters. Vector search returns its top *k* for any\n"
        "input, so an out-of-domain question still comes back with four confident-looking\n"
        "chunks; a service that simply answers from whatever it retrieved would answer\n"
        "those too. A relevance floor (IDF-weighted coverage of the question's vocabulary,\n"
        "with unknown words counting against the score) drops evidence that is not about the\n"
        "question, and when nothing survives the service declines instead of guessing.\n"
    )

    parts.append("\n## Transcript\n\n")
    for record in records:
        parts.append(f"### {record['question']}\n\n")
        parts.append(f"*{record['category']} · generator: `{record['generator']}` · "
                     f"grounded: {record['grounded']}*\n\n")
        answer = record["answer"].strip()
        parts.append("```\n" + answer + "\n```\n\n")
        if record["citations"]:
            shown = record["citations"][:6]
            parts.append("Sources: " + ", ".join(f"`{c}`" for c in shown))
            if len(record["citations"]) > len(shown):
                parts.append(f" (+{len(record['citations']) - len(shown)} more)")
            parts.append("\n\n")

    parts.append("## Which backends actually ran\n\n")
    parts.append("| Layer | Configured | Used in this run |\n|---|---|---|\n")
    parts.append(f"| Knowledge graph | {'Neo4j AuraDB' if neo4j_credentials() else 'not configured'} | "
                 f"`{backends.get('graph', '?')}` |\n")
    parts.append(f"| Vector store | Chroma (local, no credentials) | "
                 f"`{backends.get('documents', '?')}` |\n")
    parts.append(f"| Answer generation | {'Claude API' if anthropic_available() else 'not configured'} | "
                 f"`{backends.get('llm', 'none')}` |\n")
    parts.append(
        "\nTwo of these need credentials this machine does not have, so the run above used\n"
        "the fallbacks: the graph was served from an in-process NetworkX build of the same\n"
        "document, and answers were composed directly from retrieved evidence rather than\n"
        "phrased by Claude. Both paths are real code exercised by the tests, and the roadmap\n"
        "already nominates the in-memory graph as the Week 7 fallback.\n\n"
        "What this does **not** demonstrate is AuraDB's Cypher path or Claude's phrasing\n"
        "under load. `src/knowledge_graph.py` contains the Cypher for every query and\n"
        "`src/llm_service.py` the Claude call; adding credentials to `.env` switches both\n"
        "with no code change. Until someone runs it that way, that is a claim about the\n"
        "code, not a measured result.\n"
    )

    parts.append("\n## Read-only boundary\n\n")
    parts.append(
        "CLAUDE.md: *\"Never let the LLM service write into the RL/sim control loop.\"*\n\n"
        "`test_week7.py` imports `llm_service` in a clean subprocess and inspects\n"
        "`sys.modules`, asserting that `traci`, `libsumo`, `sumo_rl`, `ray`,\n"
        "`stable_baselines3`, `torch` and every training/eval module are absent. The service's\n"
        "public surface is `ask` and `close` — there is no method that writes anywhere. The\n"
        "boundary is verified, not asserted.\n"
    )

    parts.append("\n## Reproducing\n\n```powershell\n")
    parts.append("python src\\knowledge_graph.py    # rebuild the graph from the network + CSVs\n")
    parts.append("python src\\vector_store.py       # embed the reports into Chroma\n")
    parts.append("python src\\test_week7.py         # 9 checks: fidelity, grounding, refusal, boundary\n")
    parts.append("python src\\week7_demo.py         # regenerate this report\n")
    parts.append("python src\\llm_service.py \"which junctions feed C2?\"\n")
    parts.append("```\n")

    os.makedirs(os.path.dirname(WEEK7_REPORT), exist_ok=True)
    with open(WEEK7_REPORT, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    log.info("Wrote %s (DoD %s)", WEEK7_REPORT, "MET" if met else "NOT MET")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the Week 7 grounded-QA demonstration.")
    parser.add_argument("--no-llm", action="store_true", help="Force the extractive answerer.")
    args = parser.parse_args()
    records = run(use_llm=not args.no_llm)
    write_report(records)


if __name__ == "__main__":
    main()
