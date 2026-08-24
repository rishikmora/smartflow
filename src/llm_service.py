"""Read-only question answering grounded in the corridor graph and the project reports.

Week 7's Definition of Done is that "a natural-language query returns an answer
grounded in real graph data, not a hallucinated guess". That is a claim about
*provenance*, so this service is built so provenance is structural rather than
promised:

1. **Retrieve first, always.** The question is parsed for entities the graph knows
   about — junction ids, controller names, metrics, scenarios, rules — and each match
   dispatches a typed graph query. Document passages are retrieved in parallel.
2. **Refuse when nothing is retrieved.** If neither source yields evidence the service
   says so and stops. It never falls through to the model's own knowledge, which is the
   single most common way a RAG system starts hallucinating.
3. **Answer only from the retrieved context.** With Claude credentials the model is
   given the evidence and forbidden from adding anything outside it. Without them the
   answer is composed directly from the same facts. Either way the returned object
   carries the exact facts and passages behind it, so any sentence can be checked.

Every answer records ``generator`` and ``backends`` so a reader can see whether Claude,
AuraDB and Chroma actually ran, or whether a fallback did.

The read-only boundary
----------------------
CLAUDE.md draws a hard line: the LLM service must never write into the RL/sim control
loop. This module imports only the graph and the document store — nothing that can
start SUMO, construct an environment, or touch a policy — and ``test_week7.py``
verifies that by inspecting the transitive import graph rather than trusting review.

Usage:
    python src/llm_service.py "which junctions feed C2?"
    python src/llm_service.py --no-llm "what is the minimum green time?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import GraphQueries, open_graph
from vector_store import DocumentStore, open_store
from week7_config import (
    ANTHROPIC_EFFORT,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    RELEVANCE_THRESHOLD,
    TOP_K_DOCS,
    anthropic_available,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# Words a person would use, mapped to the metric names the graph stores.
METRIC_SYNONYMS = {
    "avg_wait_time_s": ["wait", "waiting", "delay", "delayed"],
    "throughput_veh": ["throughput", "arrived", "arrivals", "completed", "cleared"],
    "max_queue_len": ["queue", "queues", "backlog", "jam"],
}
SCENARIOS = ["base", "light", "peak", "asymmetric"]
RULE_SYNONYMS = {
    "min_green_s": ["minimum green", "min green"],
    "max_green_s": ["maximum green", "max green"],
    "yellow_time_s": ["yellow", "amber", "clearance"],
    "delta_time_s": ["decision", "delta time", "act", "acts", "how often"],
    "episode_s": ["episode", "how long", "duration"],
}

SYSTEM_PROMPT = """You are SmartFlow's analytics assistant. You answer questions about \
one specific traffic-simulation project using ONLY the evidence supplied in the user \
message.

Hard rules:
- Use only the GRAPH FACTS and DOCUMENT PASSAGES given. They are the complete evidence.
- Never add numbers, names or claims from your own knowledge of traffic engineering, \
reinforcement learning, or SUMO. If the evidence does not contain something, say it is \
not in the available data.
- Do not soften or round the numbers you are given; quote them as supplied.
- Be direct and concise: two to five sentences. No preamble.
- You are read-only. You never propose changing the running simulation or controller.
"""


@dataclass
class Fact:
    """One retrieved statement plus where it came from."""

    text: str
    source: str
    kind: str = "graph"


@dataclass
class GroundedAnswer:
    """An answer together with everything it was derived from."""

    question: str
    answer: str
    grounded: bool
    generator: str
    backends: dict[str, str]
    facts: list[Fact] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)

    def citations(self) -> list[str]:
        """Return every distinct evidence id behind this answer."""
        out = [f.source for f in self.facts]
        out += [p.get("chunk_id", "?") for p in self.passages]
        return sorted(set(out))

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the QA transcript."""
        data = asdict(self)
        data["citations"] = self.citations()
        return data


# ── entity extraction ───────────────────────────────────────────────────────
def _mentions(text: str, term: str, whole_word: bool = False) -> bool:
    """Test whether a term appears in text as a word, not as a substring.

    A plain ``term in text`` test is wrong here and fails in ways that are hard to
    spot: "rl" matches inside "wo**rl**d", so "who won the 2019 cricket world cup?"
    retrieved every benchmark result in the graph and the service answered it. "act"
    matching inside "exactly" is the same bug waiting to happen.

    Short terms are anchored at both ends; longer ones only at the start, so "wait"
    still matches "waiting" and "queue" matches "queues".

    Args:
        text: lower-cased haystack.
        term: term or phrase to look for.
        whole_word: force a trailing boundary regardless of length.

    Returns:
        Whether the term occurs as a word.
    """
    pattern = r"\b" + re.escape(term)
    if whole_word or len(term) <= 3:
        pattern += r"\b"
    return re.search(pattern, text) is not None


def _find_junctions(question: str, known: list[str]) -> list[str]:
    """Return junction ids mentioned in the question."""
    candidates = set(re.findall(r"\b([A-Da-d][0-3])\b", question))
    upper = {c.upper() for c in candidates}
    return sorted(j for j in known if j in upper)


def _find_controllers(question: str, known: list[str]) -> list[str]:
    """Return controller ids mentioned in the question, matching loosely."""
    lowered = question.lower()
    hits = {c for c in known if _mentions(lowered, c.lower())}
    # allow the human names people actually type
    if _mentions(lowered, "fixed"):
        hits |= {c for c in known if c == "fixed"}
    if _mentions(lowered, "actuated"):
        hits |= {c for c in known if c == "actuated"}
    if any(_mentions(lowered, w, whole_word=True)
           for w in ("rl", "reinforcement", "smartflow", "marl", "shared policy")):
        hits |= {c for c in known if c.startswith("marl")}
    return sorted(hits)


def _find_metrics(question: str) -> list[str]:
    """Return metric names implied by the question's wording."""
    lowered = question.lower()
    return sorted(
        metric for metric, words in METRIC_SYNONYMS.items()
        if any(_mentions(lowered, w) for w in words)
    )


def _find_scenarios(question: str) -> list[str]:
    """Return demand scenarios mentioned in the question."""
    lowered = question.lower()
    # whole-word: "base" must not match "based", which appears throughout the reports
    return [s for s in SCENARIOS if _mentions(lowered, s, whole_word=True)]


def _find_rules(question: str) -> list[str]:
    """Return rule ids implied by the question's wording."""
    lowered = question.lower()
    return sorted(
        rule for rule, words in RULE_SYNONYMS.items()
        if any(_mentions(lowered, w) for w in words)
    )


# ── graph retrieval ─────────────────────────────────────────────────────────
def retrieve_graph_facts(question: str, graph: GraphQueries, limit: int = 24) -> list[Fact]:
    """Query the knowledge graph for everything relevant to a question.

    Args:
        question: the natural-language question.
        graph: an open graph backend.
        limit: maximum facts to return.

    Returns:
        Facts with a ``graph:`` provenance string each.
    """
    facts: list[Fact] = []
    lowered = question.lower()
    signalised = graph.signalised_junctions()
    junctions = _find_junctions(question, [j for j in signalised] + signalised)

    # network-wide questions
    if any(w in lowered for w in ("how many junction", "how many intersection",
                                  "how many signal", "network", "corridor look",
                                  "how big", "topology")):
        stats = graph.stats()
        facts.append(Fact(
            f"The corridor graph holds {stats['nodes']} nodes and {stats['relationships']} "
            f"relationships: " + ", ".join(f"{v} {k}" for k, v in stats["labels"].items()) + ".",
            "graph:stats",
        ))
        facts.append(Fact(
            f"{len(signalised)} of the junctions are signalised: {', '.join(signalised)}.",
            "graph:Junction:signalised",
        ))

    for junction in junctions:
        node = graph.junction(junction)
        if node:
            facts.append(Fact(
                f"Junction {junction} sits at x={node['x']}, y={node['y']} and is "
                f"{'signalised' if node.get('signalised') else 'not signalised'} "
                f"with {node.get('incoming', '?')} incoming and {node.get('outgoing', '?')} outgoing roads.",
                f"graph:Junction:{junction}",
            ))
        neighbours = graph.neighbours(junction)
        if neighbours:
            facts.append(Fact(
                f"Junction {junction} feeds directly into {', '.join(neighbours)}.",
                f"graph:FEEDS:{junction}",
            ))
        if any(w in lowered for w in ("lane", "lanes", "approach", "approaches")):
            lanes = graph.lanes_of(junction)
            if lanes:
                facts.append(Fact(
                    f"Junction {junction} has {len(lanes)} incoming lanes: "
                    + ", ".join(l["id"] for l in lanes) + ".",
                    f"graph:Lane:{junction}",
                ))
        if any(w in lowered for w in ("sensor", "detector", "detect", "loop")):
            sensors = graph.sensors_of(junction)
            if sensors:
                facts.append(Fact(
                    f"Junction {junction} is monitored by {len(sensors)} induction-loop sensors "
                    f"(one per controlled lane). These are inferred: SUMO instantiates them "
                    f"implicitly for actuated control rather than declaring them in the network file.",
                    f"graph:Sensor:{junction}",
                ))
        if any(w in lowered for w in ("phase", "program", "cycle", "timing", "green")):
            program = graph.program_of(junction)
            if program:
                phases = program.get("phases", [])
                detail = "; ".join(
                    f"phase {p['index']} {p['kind']} {p['duration_s']}s" for p in phases
                )
                facts.append(Fact(
                    f"Junction {junction} runs a {program['type']} program with "
                    f"{program['num_phases']} phases and a {program['cycle_s']}s cycle: {detail}.",
                    f"graph:Program:{junction}",
                ))

    # rules
    for rule_id in _find_rules(question):
        for rule in graph.rules():
            if rule["id"] == rule_id:
                facts.append(Fact(
                    f"{rule['id']} = {rule['value']}{rule['unit']} - {rule['description']} "
                    f"(defined in {rule['source']}).",
                    f"graph:Rule:{rule_id}",
                ))

    # benchmark results
    controllers = _find_controllers(question, sorted({
        r["controller"] for r in graph.results()
    }))
    metrics = _find_metrics(question)
    scenarios = _find_scenarios(question) or (["base"] if (controllers or metrics) else [])
    if controllers or metrics:
        for scenario in scenarios or [None]:
            for controller in controllers or [None]:
                for metric in metrics or [None]:
                    for row in graph.results(controller=controller, scenario=scenario, metric=metric):
                        facts.append(Fact(
                            f"{row['controller']} on {row['scenario']} demand: "
                            f"{row['metric']} = {row['mean']} (std {row['std']}, "
                            f"{row['seeds']} seeds).",
                            f"graph:Result:{row['id']}",
                        ))

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique: list[Fact] = []
    for fact in facts:
        if fact.source in seen:
            continue
        seen.add(fact.source)
        unique.append(fact)
    return unique[:limit]


# ── answering ───────────────────────────────────────────────────────────────
def _format_evidence(facts: list[Fact], passages: list[dict[str, Any]]) -> str:
    """Render retrieved evidence into the block handed to the model."""
    lines = ["GRAPH FACTS (structured, from the corridor knowledge graph):"]
    lines += [f"- [{f.source}] {f.text}" for f in facts] or ["- (none)"]
    lines.append("")
    lines.append("DOCUMENT PASSAGES (from this project's own reports):")
    if passages:
        for passage in passages:
            lines.append(f"- [{passage['chunk_id']}] ({passage['source']} :: {passage['heading']})")
            lines.append(f"  {passage['text'][:700]}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _extractive_answer(question: str, facts: list[Fact], passages: list[dict[str, Any]]) -> str:
    """Compose an answer directly from retrieved evidence, with no model involved.

    Used when Claude credentials are absent. It cannot hallucinate because it only
    re-states retrieved material, at the cost of reading like a briefing rather than
    prose.

    Args:
        question: the original question.
        facts: graph facts.
        passages: retrieved document chunks.

    Returns:
        The composed answer.
    """
    parts: list[str] = []
    if facts:
        parts.append("From the corridor knowledge graph:")
        parts += [f"  - {f.text}" for f in facts[:6]]

    # Passages reaching here already cleared the relevance floor, so the best one is
    # worth quoting.
    if passages:
        best = passages[0]
        snippet = " ".join(best["text"].split())
        if len(snippet) > 420:
            snippet = snippet[:420].rsplit(" ", 1)[0] + "..."
        parts.append(f"From {best['source']} ({best['heading']}):")
        parts.append(f"  {snippet}")
    elif not facts and passages:
        # nothing shares terms but the graph had nothing either: show the best hit
        # rather than pretend there was no evidence at all
        best = passages[0]
        snippet = " ".join(best["text"].split())[:420]
        parts.append(f"Closest match in {best['source']} ({best['heading']}):")
        parts.append(f"  {snippet}")
    return "\n".join(parts)


def _claude_answer(question: str, evidence: str) -> str | None:
    """Ask Claude to phrase an answer strictly from the supplied evidence.

    Args:
        question: the user's question.
        evidence: the formatted evidence block.

    Returns:
        The answer text, or ``None`` if the API is unavailable or errored — the caller
        then falls back to the extractive path rather than failing the query.
    """
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; using the extractive answerer.")
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": ANTHROPIC_EFFORT},
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\n{evidence}\n\n"
                           "Answer using only the evidence above.",
            }],
        )
        if response.stop_reason == "refusal":
            log.warning("Claude declined the request; using the extractive answerer.")
            return None
        return "".join(b.text for b in response.content if b.type == "text").strip() or None
    except anthropic.AuthenticationError:
        log.warning("Claude credentials rejected; using the extractive answerer.")
    except anthropic.RateLimitError:
        log.warning("Claude rate-limited; using the extractive answerer.")
    except anthropic.APIStatusError as exc:
        log.warning("Claude API error %s; using the extractive answerer.", exc.status_code)
    except anthropic.APIConnectionError:
        log.warning("Could not reach the Claude API; using the extractive answerer.")
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unexpected Claude failure (%s); using the extractive answerer.", exc)
    return None


class AnalyticsService:
    """Read-only question answering over the graph and the document store."""

    def __init__(self, graph: GraphQueries, store: DocumentStore, use_llm: bool = True) -> None:
        """Wire the service to already-open backends.

        Args:
            graph: an open knowledge-graph backend.
            store: an open document store.
            use_llm: allow calls to the Claude API. ``False`` forces the extractive path.
        """
        self.graph = graph
        self.store = store
        self.use_llm = use_llm

    def ask(self, question: str, k: int = TOP_K_DOCS) -> GroundedAnswer:
        """Answer one question, grounded in retrieved evidence.

        Args:
            question: the natural-language question.
            k: number of document passages to retrieve.

        Returns:
            A :class:`GroundedAnswer`. When nothing is retrieved, ``grounded`` is
            ``False`` and the answer says so rather than guessing.
        """
        facts = retrieve_graph_facts(question, self.graph)
        retrieved = self.store.search(question, k=k)

        # Vector search always returns its top k, so "did retrieval find anything" is
        # not the same question as "is any of it about this". Passages below the
        # relevance floor are dropped entirely rather than quietly cited.
        passages = [
            {**p, "relevance": round(self.store.relevance(question, p["text"]), 3)}
            for p in retrieved
        ]
        passages = [p for p in passages if p["relevance"] >= RELEVANCE_THRESHOLD]

        backends = {
            "graph": self.graph.backend,
            "documents": self.store.backend,
            "llm": "none",
        }

        # The refusal path. This is the anti-hallucination guarantee: with no evidence
        # there is nothing to be grounded in, so the service declines instead of
        # letting a model answer from its own prior.
        if not facts and not passages:
            return GroundedAnswer(
                question=question,
                answer=("I don't have data for that. The corridor knowledge graph and the "
                        "project's reports contain nothing matching this question, and this "
                        "service only answers from those two sources."),
                grounded=False,
                generator="refusal",
                backends=backends,
                facts=[], passages=[],
            )

        evidence = _format_evidence(facts, passages)
        answer = None
        generator = "extractive"
        if self.use_llm and anthropic_available():
            answer = _claude_answer(question, evidence)
            if answer:
                generator = "claude"
                backends["llm"] = ANTHROPIC_MODEL
        if answer is None:
            answer = _extractive_answer(question, facts, passages)

        return GroundedAnswer(
            question=question, answer=answer, grounded=True, generator=generator,
            backends=backends, facts=facts, passages=passages,
        )

    def close(self) -> None:
        """Release the graph backend."""
        self.graph.close()


def open_service(use_llm: bool = True) -> AnalyticsService:
    """Open the graph, the document store and the service over them."""
    return AnalyticsService(open_graph(), open_store(), use_llm=use_llm)


def main() -> None:
    """CLI entry point: answer one question from the command line."""
    parser = argparse.ArgumentParser(description="Ask the SmartFlow analytics service a question.")
    parser.add_argument("question", nargs="+", help="Natural-language question.")
    parser.add_argument("--no-llm", action="store_true", help="Force the extractive answerer.")
    parser.add_argument("--json", action="store_true", help="Print the full grounded answer as JSON.")
    args = parser.parse_args()

    # project reports contain em dashes and emoji; a cp1252 console would crash on them
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    service = open_service(use_llm=not args.no_llm)
    try:
        result = service.ask(" ".join(args.question))
    finally:
        service.close()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"\n{result.answer}\n")
    print(f"[generator={result.generator} grounded={result.grounded} "
          f"graph={result.backends['graph']} docs={result.backends['documents']}]")
    if result.citations():
        print("sources: " + ", ".join(result.citations()[:8]))


if __name__ == "__main__":
    main()
