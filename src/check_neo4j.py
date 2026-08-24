"""Verify Neo4j credentials and load the corridor graph into AuraDB.

Run this after filling in ``.env``. It answers three questions in order and stops at
the first failure, so a problem points at one cause instead of a stack trace:

1. Are the variables present?
2. Do they connect?
3. Does the graph load and answer a real query?

**It never prints your password**, and it masks the instance id in the URI, so the
output is safe to paste into a chat or a report.

Usage:
    python src/check_neo4j.py
    python src/check_neo4j.py --reload    # wipe and re-load the graph
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week7_config import (
    ENV_FILE,
    NEO4J_PASSWORD_VAR,
    NEO4J_URI_VAR,
    NEO4J_USER_VAR,
    load_env,
    neo4j_credentials,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def _mask_uri(uri: str) -> str:
    """Hide the instance id in a connection URI so output can be shared.

    Args:
        uri: e.g. ``neo4j+s://abc12345.databases.neo4j.io``.

    Returns:
        The same URI with the instance id replaced by asterisks.
    """
    if "://" not in uri:
        return "***"
    scheme, rest = uri.split("://", 1)
    host = rest.split("/", 1)[0]
    parts = host.split(".", 1)
    masked = ("*" * len(parts[0])) + ("." + parts[1] if len(parts) > 1 else "")
    return f"{scheme}://{masked}"


def main() -> None:
    """Check credentials, connect, load the graph and run a sample query."""
    parser = argparse.ArgumentParser(description="Verify Neo4j credentials for SmartFlow.")
    parser.add_argument("--reload", action="store_true", help="Wipe and re-load the graph.")
    args = parser.parse_args()

    # ── 1. variables present? ───────────────────────────────────────────────
    load_env()
    if not os.path.isfile(ENV_FILE):
        log.error("No .env file at %s", ENV_FILE)
        log.error("Create one:  cp .env.example .env    then fill in the Neo4j values.")
        raise SystemExit(1)

    missing = [v for v in (NEO4J_URI_VAR, NEO4J_USER_VAR, NEO4J_PASSWORD_VAR)
               if not os.environ.get(v)]
    if missing:
        log.error("These are empty in .env: %s", ", ".join(missing))
        log.error("AuraDB shows all three once, at instance creation. If you lost them, "
                  "reset the password from the instance's ... menu in the console.")
        raise SystemExit(1)

    credentials = neo4j_credentials()
    assert credentials is not None
    log.info("[1/3] credentials found  uri=%s  user=%s  password=%s",
             _mask_uri(credentials["uri"]), credentials["user"],
             "*" * 8)

    # ── 2. connects? ────────────────────────────────────────────────────────
    try:
        from knowledge_graph import Neo4jGraph, load_document
    except ImportError as exc:
        log.error("Could not import the graph module: %s", exc)
        raise SystemExit(1)

    try:
        graph = Neo4jGraph(credentials)
    except Exception as exc:
        message = str(exc)
        log.error("[2/3] could not connect: %s", type(exc).__name__)
        if "authentication" in message.lower() or "unauthorized" in message.lower():
            log.error("  The username or password is wrong. Username is almost always 'neo4j'.")
        elif "resolve" in message.lower() or "name or service" in message.lower():
            log.error("  The URI looks wrong. It should start with neo4j+s:// and end with "
                      ".databases.neo4j.io")
        elif "routing" in message.lower() or "unavailable" in message.lower():
            log.error("  Reachable but not serving. A free AuraDB instance auto-pauses when "
                      "idle — open the console and press Resume, then retry.")
        else:
            log.error("  %s", message[:300])
        raise SystemExit(1)
    log.info("[2/3] connected")

    # ── 3. load and query ───────────────────────────────────────────────────
    try:
        document = load_document()
        stats = graph.stats()
        if args.reload or stats["nodes"] == 0:
            log.info("      loading %d nodes into AuraDB (this takes a minute)...",
                     len(document["nodes"]))
            graph.load(document, wipe=True)
            stats = graph.stats()

        neighbours = graph.neighbours("C2")
        rules = graph.rules()
        log.info("[3/3] graph live: %d nodes, %d relationships",
                 stats["nodes"], stats["relationships"])
        log.info("      sample Cypher: C2 feeds -> %s", ", ".join(neighbours) or "(none)")
        log.info("      %d rules loaded, e.g. %s", len(rules),
                 rules[0]["id"] if rules else "(none)")
    finally:
        graph.close()

    log.info("")
    log.info("Neo4j is wired up. From now on the service uses AuraDB automatically:")
    log.info("  python src/week7_demo.py            # backends table will say 'neo4j aura'")
    log.info("  python src/llm_service.py \"which junctions does C2 feed into?\"")


if __name__ == "__main__":
    main()
