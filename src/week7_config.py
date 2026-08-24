"""Week 7 configuration: knowledge graph, vector store and the read-only LLM service.

Week 7 adds a question-answering layer over the corridor: a knowledge graph of the
road network, its signal programs and the rules that govern them, plus a vector store
over the project's own written reports. The LLM service consults **both** before it
answers, and every claim it makes is traceable to one of them.

Three services, three credential situations
-------------------------------------------
``Neo4j AuraDB``
    Used when ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` are set. Without
    them the same graph is built in-process on NetworkX, which is the fallback the
    roadmap already nominates ("fall back to SUMO's own in-memory network graph").
    Both backends answer the identical set of typed queries, so the service above them
    cannot tell which is underneath.

``Chroma``
    Local and needs no credentials, but its default embedding function downloads a
    model on first use. If that is unavailable the store falls back to lexical BM25-ish
    retrieval over the same chunks.

``Claude API``
    Used when credentials resolve (``ANTHROPIC_API_KEY``, or an ``ant auth login``
    profile). Without them the service still answers, by composing the retrieved facts
    directly instead of generating prose.

The point of the fallbacks is not to pretend the real services ran. It is that the
*grounding* — the thing Week 7's Definition of Done is actually about — is a property
of the retrieval layer, not of which vendor answers. Every answer records which
backends produced it.

Architectural boundary
----------------------
CLAUDE.md: "Never let the LLM service write into the RL/sim control loop. Read-only
analytics and orchestration only." This module and everything downstream of it import
nothing that can start a simulation or mutate a policy, and
``test_week7.py`` asserts that boundary rather than trusting it.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
KG_DIR = os.path.join(OUTPUTS_DIR, "kg")
CHROMA_DIR = os.path.join(KG_DIR, "chroma")

CORRIDOR_NET = os.path.join(DATA_DIR, "corridor.net.xml")
CORRIDOR_ACTUATED_NET = os.path.join(DATA_DIR, "corridor_actuated.net.xml")

# ── credentials (never hardcoded; loaded from .env via python-dotenv) ────────
ENV_FILE = os.path.join(ROOT, ".env")
NEO4J_URI_VAR = "NEO4J_URI"
NEO4J_USER_VAR = "NEO4J_USERNAME"
NEO4J_PASSWORD_VAR = "NEO4J_PASSWORD"
NEO4J_DATABASE_VAR = "NEO4J_DATABASE"
ANTHROPIC_KEY_VAR = "ANTHROPIC_API_KEY"

# Per the claude-api reference: default to Opus 5 with adaptive thinking. This is an
# analytics service, so effort stays low — the reasoning is in the retrieval, not the
# generation, and the model's job is to phrase grounded facts, not to solve anything.
ANTHROPIC_MODEL = "claude-opus-5"
ANTHROPIC_MAX_TOKENS = 8000   # adaptive thinking tokens count toward this, so leave room
ANTHROPIC_EFFORT = "low"

# ── vector store ─────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "smartflow_docs"
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K_DOCS = 4

# Minimum share of a question's IDF-weighted vocabulary a passage must cover to count
# as evidence. Vector search returns its top k however weak the match, so without this
# an out-of-domain question still retrieves four confident-looking chunks and the
# service answers it. Calibrated on the real corpus: out-of-domain questions score
# 0.00, in-domain ones 0.37-0.77.
RELEVANCE_THRESHOLD = 0.25

# Documents the RAG layer indexes. These are the project's own reports, so an answer
# grounded in them is grounded in measured results rather than model recall.
RAG_SOURCES = [
    os.path.join(ROOT, "BENCHMARK_REPORT.md"),
    os.path.join(OUTPUTS_DIR, "week2_literature_note.md"),
    os.path.join(OUTPUTS_DIR, "week3_report.md"),
    os.path.join(OUTPUTS_DIR, "week4_report.md"),
    os.path.join(OUTPUTS_DIR, "week4_nonstationarity_notes.md"),
    os.path.join(OUTPUTS_DIR, "week5_report.md"),
    os.path.join(ROOT, "README.md"),
]

# ── metric sources folded into the graph as Result nodes ─────────────────────
METRICS_SOURCES = {
    "corridor": os.path.join(OUTPUTS_DIR, "marl_metrics.csv"),
    "junction": os.path.join(OUTPUTS_DIR, "week3_corridor_metrics.csv"),
}

# ── outputs ──────────────────────────────────────────────────────────────────
GRAPH_JSON = os.path.join(KG_DIR, "corridor_graph.json")
QA_JSON = os.path.join(OUTPUTS_DIR, "week7_qa.json")
WEEK7_REPORT = os.path.join(OUTPUTS_DIR, "week7_report.md")

# Simulation constants that exist as Rule nodes in the graph, with the file that is
# their source of truth. Keeping the provenance next to the value is what lets an
# answer cite where a rule came from.
SIGNAL_RULES = [
    ("min_green_s", 5, "s", "minimum green time before a phase may be cut", "week4_config.MIN_GREEN"),
    ("max_green_s", 90, "s", "maximum green time before a phase must be cut", "week4_config.MAX_GREEN"),
    ("yellow_time_s", 2, "s", "clearance interval between conflicting greens", "week4_config.YELLOW_TIME"),
    ("delta_time_s", 5, "s", "simulated seconds between agent decisions", "week4_config.DELTA_TIME"),
    ("episode_s", 1800, "s", "simulated seconds per evaluation episode", "week4_config.SIM_SECONDS"),
]


def load_env() -> None:
    """Load ``.env`` into the process environment if python-dotenv is available.

    Credentials are never committed; ``.env`` is git-ignored and ``.env.example``
    documents the variable names.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    if os.path.isfile(ENV_FILE):
        load_dotenv(ENV_FILE, override=False)


def neo4j_credentials() -> dict[str, str] | None:
    """Return Neo4j connection settings, or ``None`` when they are not configured.

    Returns:
        ``{"uri", "user", "password", "database"}`` or ``None``.
    """
    load_env()
    uri = os.environ.get(NEO4J_URI_VAR)
    user = os.environ.get(NEO4J_USER_VAR)
    password = os.environ.get(NEO4J_PASSWORD_VAR)
    if not (uri and user and password):
        return None
    return {
        "uri": uri,
        "user": user,
        "password": password,
        "database": os.environ.get(NEO4J_DATABASE_VAR, "neo4j"),
    }


def anthropic_available() -> bool:
    """Return whether Claude API credentials appear to be configured.

    An unset ``ANTHROPIC_API_KEY`` does not prove there are no credentials — the SDK
    also resolves ``ANTHROPIC_AUTH_TOKEN`` and ``ant auth login`` profiles — so this
    reports only the signals visible from the environment. The service treats a failed
    call as "unavailable" rather than relying on this alone.
    """
    load_env()
    return bool(os.environ.get(ANTHROPIC_KEY_VAR) or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
