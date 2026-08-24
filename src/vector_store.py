"""Retrieval over the project's own written reports (Chroma, with a lexical fallback).

The graph answers structured questions ("which junctions feed C2", "what is the
minimum green"). This store answers the prose ones — why a Definition of Done failed,
what the fairness constraint turned out to do, why the ablations came back identical.
Both feed the same answering layer.

What is indexed is deliberately narrow: the project's own reports and notes. That means
a retrieved passage is a claim this project already measured and wrote down, with a
file and heading to cite. Indexing the wider internet would make the service sound more
knowledgeable and be much harder to ground.

Two backends, same contract
---------------------------
Chroma with its default embedding function is the intended path, but that function
downloads an ONNX model on first use. Where that is unavailable the store falls back to
lexical scoring — IDF-weighted term overlap with a phrase bonus — over exactly the same
chunks. Retrieval quality differs; grounding does not, because both return real
passages with real provenance. :attr:`DocumentStore.backend` records which ran.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week7_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    RAG_SOURCES,
    ROOT,
    TOP_K_DOCS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9_]+")
# Terms so common in these documents that matching on them says nothing about relevance.
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "not",
    "but", "its", "it", "is", "of", "to", "in", "on", "at", "as", "by", "a", "an",
    "be", "has", "have", "had", "than", "then", "so", "which", "what", "how", "does",
    "do", "did", "can", "will", "would", "there", "their", "they", "you", "we", "or",
}


def _tokens(text: str) -> list[str]:
    """Lower-case word tokens with stopwords removed."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def chunk_markdown(text: str, source: str) -> list[dict[str, Any]]:
    """Split a Markdown document into retrievable chunks that keep their heading.

    Splitting on headings first means a chunk carries the section it came from, so a
    citation can name the heading rather than a character offset.

    Args:
        text: the document body.
        source: display name recorded on every chunk.

    Returns:
        Chunks with ``text``, ``source``, ``heading`` and ``chunk_id``.
    """
    sections: list[tuple[str, list[str]]] = []
    heading = "(intro)"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if buffer:
                sections.append((heading, buffer))
            heading = line.lstrip("#").strip() or "(untitled)"
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, buffer))

    chunks: list[dict[str, Any]] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        start = 0
        while start < len(body):
            piece = body[start:start + CHUNK_CHARS].strip()
            if len(piece) > 40:  # skip slivers left by the overlap window
                chunks.append({
                    "text": piece,
                    "source": source,
                    "heading": heading,
                    "chunk_id": f"{source}#{heading}#{len(chunks)}",
                })
            if start + CHUNK_CHARS >= len(body):
                break
            start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def collect_chunks(sources: list[str] | None = None) -> list[dict[str, Any]]:
    """Read and chunk every configured source document.

    Args:
        sources: file paths; defaults to :data:`week7_config.RAG_SOURCES`.

    Returns:
        All chunks, in source order. Missing files are warned about, not fatal — the
        reports are generated artefacts and may legitimately not exist yet.
    """
    chunks: list[dict[str, Any]] = []
    for path in sources or RAG_SOURCES:
        if not os.path.isfile(path):
            log.warning("RAG source missing, skipping: %s", path)
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        name = os.path.relpath(path, ROOT).replace("\\", "/")
        chunks.extend(chunk_markdown(text, name))
    log.info("Collected %d chunks from %d documents", len(chunks), len(sources or RAG_SOURCES))
    return chunks


class DocumentStore:
    """Semantic (or lexical) retrieval over the project's reports."""

    def __init__(self, chunks: list[dict[str, Any]], backend: str) -> None:
        """Prefer :func:`open_store`; this takes an already-selected backend."""
        self.chunks = chunks
        self.backend = backend
        self._collection: Any = None
        self._idf: dict[str, float] = {}
        self._chunk_tokens: list[Counter] = []

    # ── lexical fallback ────────────────────────────────────────────────────
    def _build_lexical(self) -> None:
        """Precompute IDF and per-chunk term counts."""
        self._chunk_tokens = [Counter(_tokens(c["text"])) for c in self.chunks]
        document_frequency: Counter = Counter()
        for counts in self._chunk_tokens:
            document_frequency.update(counts.keys())
        total = max(1, len(self._chunk_tokens))
        self._idf = {
            term: math.log(1 + total / (1 + freq))
            for term, freq in document_frequency.items()
        }

    def _lexical_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Score chunks by IDF-weighted term overlap, with a phrase bonus."""
        wanted = _tokens(query)
        if not wanted:
            return []
        lowered = query.lower()
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self._chunk_tokens):
            score = 0.0
            for term in wanted:
                if term in counts:
                    # sub-linear term frequency, so one repeated word cannot dominate
                    score += self._idf.get(term, 0.0) * (1 + math.log(counts[term]))
            # reward an exact multi-word phrase, which term overlap alone misses
            for size in (3, 2):
                for i in range(len(wanted) - size + 1):
                    phrase = " ".join(wanted[i:i + size])
                    if phrase in self.chunks[index]["text"].lower():
                        score += 2.5 * size
            if lowered.strip("?") in self.chunks[index]["text"].lower():
                score += 8
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        return [
            {**self.chunks[i], "score": round(s, 3), "retrieval": "lexical"}
            for s, i in scored[:k]
        ]

    # ── chroma ──────────────────────────────────────────────────────────────
    def _chroma_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Query the Chroma collection."""
        result = self._collection.query(query_texts=[query], n_results=k)
        out: list[dict[str, Any]] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for text, meta, distance in zip(documents, metadatas, distances):
            out.append({
                "text": text,
                "source": meta.get("source", "?"),
                "heading": meta.get("heading", "?"),
                "chunk_id": meta.get("chunk_id", "?"),
                # Chroma returns a distance; invert it so higher is better everywhere
                "score": round(1.0 / (1.0 + float(distance)), 3),
                "retrieval": "chroma",
            })
        return out

    def relevance(self, query: str, text: str) -> float:
        """Score how much of a query's *distinctive* vocabulary a passage covers.

        Vector search returns its top ``k`` no matter how weak the match, so an
        out-of-domain question ("average rainfall in Hyderabad") still comes back with
        four confident-looking chunks. Without a check like this the service would
        answer it, which is exactly the failure Week 7 is supposed to rule out.

        Terms are IDF-weighted, and a term absent from the corpus entirely is given the
        maximum weight — so unknown words like "rainfall" actively push the score down
        rather than being ignored.

        Args:
            query: the question.
            text: candidate passage.

        Returns:
            Covered IDF mass as a fraction of the query's total, in ``[0, 1]``.
        """
        terms = set(_tokens(query))
        if not terms:
            return 0.0
        if not self._idf:
            self._build_lexical()
        ceiling = max(self._idf.values(), default=1.0)
        lowered = text.lower()
        total = 0.0
        covered = 0.0
        for term in terms:
            weight = self._idf.get(term, ceiling)
            total += weight
            if term in lowered:
                covered += weight
        return covered / total if total else 0.0

    def search(self, query: str, k: int = TOP_K_DOCS) -> list[dict[str, Any]]:
        """Return the ``k`` most relevant chunks for a query.

        Args:
            query: natural-language question.
            k: number of chunks to return.

        Returns:
            Chunks with provenance and a score, highest first.
        """
        if self._collection is not None:
            try:
                return self._chroma_search(query, k)
            except Exception as exc:  # pragma: no cover - depends on local install
                log.warning("Chroma query failed (%s); using lexical retrieval.", exc)
                self._collection = None
                self.backend = "lexical (chroma query failed)"
                self._build_lexical()
        return self._lexical_search(query, k)


def open_store(prefer_chroma: bool = True, rebuild: bool = False) -> DocumentStore:
    """Build or open the document store, choosing the best available backend.

    Args:
        prefer_chroma: attempt Chroma before falling back to lexical retrieval.
        rebuild: re-embed even if the persisted collection already has content.

    Returns:
        A ready :class:`DocumentStore`.
    """
    chunks = collect_chunks()
    store = DocumentStore(chunks, backend="lexical")

    if prefer_chroma and chunks:
        try:
            import chromadb

            os.makedirs(CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = client.get_or_create_collection(CHROMA_COLLECTION)
            if rebuild or collection.count() != len(chunks):
                if collection.count():
                    client.delete_collection(CHROMA_COLLECTION)
                    collection = client.get_or_create_collection(CHROMA_COLLECTION)
                collection.add(
                    ids=[c["chunk_id"] for c in chunks],
                    documents=[c["text"] for c in chunks],
                    metadatas=[
                        {"source": c["source"], "heading": c["heading"], "chunk_id": c["chunk_id"]}
                        for c in chunks
                    ],
                )
                log.info("Embedded %d chunks into Chroma at %s", len(chunks), CHROMA_DIR)
            store._collection = collection
            store.backend = "chroma"
            # the relevance gate needs corpus IDF regardless of which backend retrieves
            store._build_lexical()
            return store
        except Exception as exc:
            log.warning("Chroma unavailable (%s); falling back to lexical retrieval.", exc)

    store._build_lexical()
    store.backend = store.backend if store.backend.startswith("lexical") else "lexical"
    return store


def main() -> None:
    """Build the store and run a couple of sample retrievals."""
    store = open_store()
    log.info("backend=%s chunks=%d", store.backend, len(store.chunks))
    for query in ("why did the fairness constraint not work",
                  "what happened at peak demand"):
        hits = store.search(query, k=2)
        log.info("query: %s", query)
        for hit in hits:
            log.info("  [%.3f] %s :: %s", hit["score"], hit["source"], hit["heading"])


if __name__ == "__main__":
    main()
