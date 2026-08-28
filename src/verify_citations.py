"""Verify every citation against the registry that issued its identifier.

Week 12 asks that every citation's DOI be checked directly against the primary
source. Publisher pages (IEEE Xplore, ACM) block automated fetching, so this
resolves each identifier against the registry that actually issued it:

* **CrossRef** (``api.crossref.org``) for DOIs — the registration agency's own
  record, which is what a DOI *is*. If CrossRef does not have it, the DOI is not
  registered, whatever a bibliography claims.
* **arXiv** (``export.arxiv.org``) for preprints, which have no DOI of their own
  but a stable, citable identifier.

For each entry the registered title is compared against the title this project
claims, so a transposed DOI pointing at a real but different paper is caught -
which is the failure mode that matters. A DOI that resolves is not the same as a
DOI that resolves *to the thing you cited*.

Entries with no identifier - software released only on GitHub - are reported as
such rather than given a fabricated one.

Usage:
    python src/verify_citations.py
    python src/verify_citations.py --offline    # re-render from the cache
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
REPORT = os.path.join(OUTPUTS, "citations_verified.md")
CACHE = os.path.join(OUTPUTS, "citations_cache.json")

USER_AGENT = ("SmartFlow-citation-check/1.0 (B.Tech project; mailto:noreply@example.com)")
TITLE_MATCH_THRESHOLD = 0.82

# What this project actually depends on, and why. Anything not used is not here:
# a bibliography padded with unread papers is worse than a short honest one.
CITATIONS: list[dict[str, Any]] = [
    {
        "key": "sumo",
        "title": "Microscopic Traffic Simulation using SUMO",
        "doi": "10.1109/ITSC.2018.8569938",
        "used_for": "The simulator every result in this project is measured in.",
    },
    {
        "key": "ppo",
        "title": "Proximal Policy Optimization Algorithms",
        "arxiv": "1707.06347",
        "used_for": "The RL algorithm used in Weeks 2-6 and 9.",
    },
    {
        "key": "sb3",
        "title": "Stable-Baselines3: Reliable Reinforcement Learning Implementations",
        "url": "http://jmlr.org/papers/v22/20-1364.html",
        "used_for": "The PPO implementation for single-agent work (Weeks 2-3, 9).",
        "note": "JMLR 22(268):1-8, 2021. JMLR assigns no DOIs, so this is verified "
                "against the journal's own article page. An arXiv identifier was "
                "guessed for this entry first and resolved to an unrelated paper - "
                "which is exactly what this script exists to catch.",
    },
    {
        "key": "rllib",
        "title": "RLlib: Abstractions for Distributed Reinforcement Learning",
        "arxiv": "1712.09381",
        "used_for": "The multi-agent training stack for Weeks 4-6.",
    },
    {
        "key": "gat",
        "title": "Graph Attention Networks",
        "arxiv": "1710.10903",
        "used_for": "The graph-attention state encoder ablated in Weeks 4-6.",
    },
    {
        "key": "fedavg",
        "title": "Communication-Efficient Learning of Deep Networks from Decentralized Data",
        "arxiv": "1602.05629",
        "used_for": "The averaging algorithm behind Week 9's federated experiment.",
    },
    {
        "key": "flower",
        "title": "Flower: A Friendly Federated Learning Research Framework",
        "arxiv": "2007.14390",
        "used_for": "The federated learning framework used in Week 9.",
    },
    {
        "key": "lora",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "arxiv": "2106.09685",
        "used_for": "The adapter method used in Week 9's fine-tune.",
    },
    {
        "key": "ua-detrac",
        "title": "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking",
        "doi": "10.1016/j.cviu.2020.102907",
        "used_for": "The vehicle-detection benchmark Week 8 was meant to use, and did not.",
    },
    {
        "key": "sumo-rl",
        "title": "SUMO-RL",
        "software": "https://github.com/LucasAlegre/sumo-rl",
        "used_for": "The Gymnasium/PettingZoo environment wrapper used from Week 2 onward.",
        "note": "Software with no DOI or preprint. Cited by repository, not fabricated.",
    },
    {
        "key": "ultralytics",
        "title": "Ultralytics YOLOv8",
        "software": "https://github.com/ultralytics/ultralytics",
        "used_for": "The detector architecture and training loop in Week 8.",
        "note": "Software with no DOI or accompanying paper for v8.",
    },
]


def _get(url: str) -> str | None:
    """Fetch a URL, returning None on any failure.

    Args:
        url: the URL to fetch.

    Returns:
        The response body, or None.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        log.warning("  fetch failed: %s", str(exc)[:120])
        return None


def resolve_doi(doi: str) -> dict[str, Any] | None:
    """Look a DOI up in CrossRef's registry.

    Args:
        doi: the DOI, without a URL prefix.

    Returns:
        ``{"title", "year", "authors", "venue"}``, or None if unregistered.
    """
    body = _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not body:
        return None
    try:
        message = json.loads(body)["message"]
    except (ValueError, KeyError):
        return None
    authors = [f"{a.get('family', '')}".strip()
               for a in message.get("author", []) if a.get("family")]
    date = (message.get("issued", {}).get("date-parts") or [[None]])[0]
    return {
        "title": (message.get("title") or [""])[0],
        "year": date[0] if date else None,
        "authors": authors,
        "venue": (message.get("container-title") or [""])[0],
        "type": message.get("type"),
    }


def resolve_arxiv(identifier: str) -> dict[str, Any] | None:
    """Look an arXiv identifier up in the arXiv API.

    Args:
        identifier: e.g. ``"1707.06347"``.

    Returns:
        ``{"title", "year", "authors", "venue"}``, or None if not found.
    """
    body = _get("http://export.arxiv.org/api/query?id_list=" + identifier)
    if not body:
        return None
    title = re.search(r"<entry>.*?<title>(.*?)</title>", body, re.S)
    published = re.search(r"<entry>.*?<published>(\d{4})", body, re.S)
    authors = re.findall(r"<author>\s*<name>(.*?)</name>", body, re.S)
    if not title:
        return None
    return {
        "title": re.sub(r"\s+", " ", title.group(1)).strip(),
        "year": int(published.group(1)) if published else None,
        "authors": [a.split()[-1] for a in authors],
        "venue": "arXiv preprint",
        "type": "preprint",
    }


def resolve_url(url: str, expected_title: str) -> dict[str, Any] | None:
    """Verify a citation against the publisher's own article page.

    Used for journals that assign no DOIs - JMLR being the one this project
    needs. The page is fetched and searched for the cited title.

    Args:
        url: the article page.
        expected_title: the title to look for.

    Returns:
        A record if the title appears on the page, otherwise None.
    """
    body = _get(url)
    if not body:
        return None
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text)
    needle = re.sub(r"\s+", " ", expected_title).strip()
    if needle.lower() not in text.lower():
        return None
    year = re.search(r"(19|20)\d{2}", text)
    return {
        "title": needle,
        "year": int(year.group(0)) if year else None,
        "authors": [],
        "venue": urllib.parse.urlparse(url).netloc,
        "type": "journal-article",
    }


def titles_match(claimed: str, registered: str) -> tuple[bool, float]:
    """Compare a claimed title against the registered one.

    Args:
        claimed: the title this project cites.
        registered: the title the registry holds.

    Returns:
        ``(matches, similarity)``.
    """
    def normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()

    ratio = difflib.SequenceMatcher(None, normalise(claimed),
                                    normalise(registered)).ratio()
    return ratio >= TITLE_MATCH_THRESHOLD, ratio


def verify(entry: dict[str, Any]) -> dict[str, Any]:
    """Verify one citation against its issuing registry.

    Args:
        entry: a citation record.

    Returns:
        The record with verification fields added.
    """
    result = dict(entry)
    if entry.get("software"):
        result["status"] = "software"
        result["detail"] = "No DOI or preprint exists; cited by repository URL."
        return result

    registered = None
    if entry.get("doi"):
        registered = resolve_doi(entry["doi"])
        source = "CrossRef"
    elif entry.get("arxiv"):
        registered = resolve_arxiv(entry["arxiv"])
        source = "arXiv"
    elif entry.get("url"):
        registered = resolve_url(entry["url"], entry["title"])
        source = "publisher page"
    else:
        result["status"] = "no-identifier"
        result["detail"] = "No DOI or arXiv identifier recorded."
        return result

    if registered is None:
        result["status"] = "unresolved"
        result["detail"] = f"{source} did not return a record for this identifier."
        return result

    matched, ratio = titles_match(entry["title"], registered["title"])
    result["registered"] = registered
    result["similarity"] = round(ratio, 3)
    result["source"] = source
    if matched:
        result["status"] = "verified"
        result["detail"] = (f"{source} record matches the cited title "
                            f"(similarity {ratio:.2f}).")
    else:
        result["status"] = "mismatch"
        result["detail"] = (f"{source} resolves this identifier to "
                            f"{registered['title']!r}, which is not the cited work "
                            f"(similarity {ratio:.2f}).")
    return result


def render(results: list[dict[str, Any]]) -> str:
    """Render the verification report.

    Args:
        results: verified citation records.

    Returns:
        Markdown.
    """
    verified = [r for r in results if r["status"] == "verified"]
    software = [r for r in results if r["status"] == "software"]
    problems = [r for r in results if r["status"] in {"mismatch", "unresolved",
                                                      "no-identifier"}]

    lines = [
        "# Citation Verification\n",
        "Every identifier below was resolved against the registry that issued it "
        "and the **registered title compared against the title this project "
        "cites**. A DOI that resolves is not the same as a DOI that resolves to "
        "the work you meant; only the second is worth anything in a bibliography.\n",
        f"Checked by `src/verify_citations.py` on "
        f"{time.strftime('%Y-%m-%d')}. "
        f"**{len(verified)} verified, {len(software)} software without an "
        f"identifier, {len(problems)} unresolved or mismatched.**\n",
        "Publisher pages (IEEE Xplore, ScienceDirect) block automated fetching, "
        "so DOIs are checked against CrossRef — the registration agency's own "
        "record, which is what a DOI actually is — and preprints against the "
        "arXiv API.\n",
        "## Verified\n",
        "| Key | Cited title | Identifier | Registered title | Year | Match |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in verified:
        identifier = (row.get("doi") or
                      ("arXiv:" + row["arxiv"] if row.get("arxiv") else row.get("url", "")))
        reg = row["registered"]
        lines.append(
            f"| `{row['key']}` | {row['title']} | `{identifier}` | "
            f"{reg['title']} | {reg['year'] or '?'} | {row['similarity']:.2f} |")

    lines.append("\n### What each is used for\n")
    for row in results:
        lines.append(f"- **`{row['key']}`** — {row['used_for']}")
        if row.get("note"):
            lines.append(f"  - {row['note']}")

    if software:
        lines.append("\n## Software cited without an identifier\n")
        lines.append("These have no DOI and no accompanying paper. They are cited "
                     "by repository rather than given a fabricated identifier.\n")
        lines.append("| Key | Project | Repository |")
        lines.append("|---|---|---|")
        for row in software:
            lines.append(f"| `{row['key']}` | {row['title']} | {row['software']} |")

    if problems:
        lines.append("\n## Unresolved or mismatched\n")
        lines.append("**These must not be cited until fixed.**\n")
        lines.append("| Key | Cited title | Identifier | Problem |")
        lines.append("|---|---|---|---|")
        for row in problems:
            identifier = (row.get("doi") or row.get("arxiv")
                          or row.get("url") or "(none)")
            lines.append(f"| `{row['key']}` | {row['title']} | `{identifier}` | "
                         f"{row['detail']} |")
    else:
        lines.append("\n## Unresolved or mismatched\n")
        lines.append("None. Every identifier resolved to the work it is cited as.\n")

    lines.append("\n## Reproducing this\n")
    lines.append("```\npython src/verify_citations.py\n```\n")
    lines.append("The script queries CrossRef and arXiv live, compares each "
                 "registered title against the cited one, and rewrites this file. "
                 "It is not a transcription of a manual check, so it cannot go "
                 "stale silently.\n")
    return "\n".join(lines)


def main() -> None:
    """Verify every citation and write the report."""
    parser = argparse.ArgumentParser(description="Verify citations against registries.")
    parser.add_argument("--offline", action="store_true",
                        help="Re-render from the cached results.")
    args = parser.parse_args()

    os.makedirs(OUTPUTS, exist_ok=True)

    if args.offline and os.path.isfile(CACHE):
        with io.open(CACHE, encoding="utf-8") as handle:
            results = json.load(handle)
        log.info("Re-rendering from cache (%d entries)", len(results))
    else:
        results = []
        for entry in CITATIONS:
            log.info("Checking %s ...", entry["key"])
            result = verify(entry)
            log.info("  %-12s %s", result["status"], result["detail"])
            results.append(result)
            time.sleep(0.6)          # be polite to the public APIs
        with io.open(CACHE, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    with io.open(REPORT, "w", encoding="utf-8") as handle:
        handle.write(render(results))

    verified = sum(1 for r in results if r["status"] == "verified")
    problems = [r for r in results if r["status"] in {"mismatch", "unresolved"}]
    log.info("")
    log.info("Wrote %s", REPORT)
    log.info("%d verified, %d software, %d problems",
             verified, sum(1 for r in results if r["status"] == "software"),
             len(problems))
    for row in problems:
        log.error("  %s: %s", row["key"], row["detail"])


if __name__ == "__main__":
    main()
