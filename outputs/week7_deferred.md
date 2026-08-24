# Week 7 — Deferred Items

**Status: nothing deferred.** Every Week 7 component runs against its real backend.

This file previously listed Neo4j, Chroma and the Claude API as outstanding. All
three were scaffolded first and wired up afterwards, so the list is now closed:

| Item | Was | Now |
|---|---|---|
| Neo4j AuraDB | needs external account credentials | live — 301 nodes / 474 relationships, queried over Cypher |
| Chroma RAG | needs `chromadb` and document ingestion | live — 101 chunks from 7 project documents |
| Claude analytics | needs API credentials in `.env` | live — `claude-opus-5` phrases every in-domain answer |

Verify any of this yourself:

```powershell
python src\check_neo4j.py      # masks the URI, never prints the password
python src\week7_demo.py       # regenerates week7_report.md with a backends table
python src\test_week7.py       # 9 checks, including the read-only boundary
```

The credential-free fallbacks (in-process NetworkX graph, extractive answerer)
were not removed. They are still exercised by `src/test_week7.py`, so the service
degrades instead of breaking on a machine without a `.env`.

## The one constraint that remains

The read-only boundary from CLAUDE.md — *"Never let the LLM service write into the
RL/sim control loop"* — is a permanent design constraint, not a deferred task. It
is enforced by `test_llm_service_cannot_touch_the_control_loop`, which imports the
service in a clean subprocess and fails if any simulation or RL module appears in
`sys.modules`.
