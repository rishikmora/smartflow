"""Build the self-contained corridor visualiser page from recorded SUMO replays.

Takes the replay files written by ``viz_recorder.py`` plus the committed 3-seed
benchmark numbers, and inlines both into ``src/viz/template.html``. The output is one
HTML file with no external requests, because the Artifact sandbox blocks them and a
demo that needs a server is a demo that can fail in front of an examiner.

The headline figures shown in the "final result" card come from
``outputs/marl_metrics.csv`` (3-seed means), *not* from the single recorded episode —
the replay shows one seed, but the claim is the 3-seed average, and the page should not
quietly imply otherwise.

Usage:
    python src/viz_build.py
    python src/viz_build.py --scenario peak --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import aggregate, load_rows
from week4_config import OUTPUTS_DIR, ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

VIZ_DIR = os.path.join(OUTPUTS_DIR, "viz")
TEMPLATE = os.path.join(ROOT, "src", "viz", "template.html")
OUTPUT = os.path.join(VIZ_DIR, "corridor_control_room.html")
MARL_CSV = os.path.join(OUTPUTS_DIR, "marl_metrics.csv")

# replay key -> (filename stem, CSV controller label)
RUN_SPEC = {
    "fixed": ("fixed", "fixed"),
    "actuated": ("actuated", "actuated"),
    "rl": ("marl_shared_w5", "marl_shared_w5"),
}


def load_replay(stem: str, scenario: str, seed: int) -> dict[str, Any]:
    """Load one recorded replay.

    Args:
        stem: filename stem, e.g. ``"fixed"``.
        scenario: demand scenario recorded.
        seed: seed recorded.

    Returns:
        The parsed replay document.

    Raises:
        FileNotFoundError: if the replay has not been recorded yet.
    """
    path = os.path.join(VIZ_DIR, f"{stem}_{scenario}_seed{seed}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing replay {path}. Record it first, e.g.\n"
            f"  python src/viz_recorder.py --controller fixed --scenario {scenario}"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def final_numbers(scenario: str) -> dict[str, dict[str, Any]]:
    """Return the committed 3-seed benchmark means for the card in the rail.

    Falls back to each replay's own single-episode summary only if the metrics CSV is
    unavailable, and says so in the log rather than silently passing off one seed as
    three.

    Args:
        scenario: demand scenario to read.

    Returns:
        ``{run key: {"wait", "thru", "queue"}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        rows = load_rows(MARL_CSV)
    except FileNotFoundError:
        log.warning("%s not found — the final-result card will be left empty.", MARL_CSV)
        return out

    summary = aggregate(
        rows, ["avg_wait_time_s", "max_queue_len", "throughput_veh"],
        scope="corridor", scenario=scenario,
    )
    for key, (_stem, label) in RUN_SPEC.items():
        stats = summary.get(label)
        if not stats:
            log.warning("No 3-seed rows for controller '%s' in scenario '%s'.", label, scenario)
            continue
        out[key] = {
            "wait": round(float(stats["avg_wait_time_s"]["mean"]), 1),
            "queue": int(round(float(stats["max_queue_len"]["mean"]))),
            "thru": int(round(float(stats["throughput_veh"]["mean"]))),
        }
    return out


def build(scenario: str = "base", seed: int = 0) -> str:
    """Assemble the visualiser page.

    Args:
        scenario: demand scenario whose replays to inline.
        seed: seed whose replays to inline.

    Returns:
        Path to the written HTML.

    Raises:
        FileNotFoundError: if the template or any replay is missing.
        ValueError: if the template has no data placeholder.
    """
    if not os.path.isfile(TEMPLATE):
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    runs = {key: load_replay(stem, scenario, seed) for key, (stem, _) in RUN_SPEC.items()}

    lengths = {key: len(run["frames"]) for key, run in runs.items()}
    if len(set(lengths.values())) != 1:
        log.warning("Replays have different frame counts %s; the player uses the shortest.", lengths)
        shortest = min(lengths.values())
        for run in runs.values():
            del run["frames"][shortest:]

    # The network and signal geometry are identical across controllers, so they are
    # kept once on the fixed run and stripped from the others.
    for key, run in runs.items():
        if key != "fixed":
            run.pop("network", None)
            run["tls"] = runs["fixed"]["tls"]

    payload = {"runs": runs, "final": final_numbers(scenario)}
    data = json.dumps(payload, separators=(",", ":"))
    # a literal </script> inside the JSON block would close the tag early
    data = data.replace("</", "<\\/")

    with open(TEMPLATE, encoding="utf-8") as handle:
        html = handle.read()
    if "/*__REPLAY_DATA__*/" not in html:
        raise ValueError("Template is missing the /*__REPLAY_DATA__*/ placeholder.")
    html = html.replace("/*__REPLAY_DATA__*/", data)

    os.makedirs(VIZ_DIR, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(html)

    size = os.path.getsize(OUTPUT) / 1024 / 1024
    log.info("Wrote %s (%.2f MB, %d frames per run)", OUTPUT, size, lengths["fixed"])
    if size > 15:
        log.warning("Page is %.1f MB — close to the 16 MB artifact limit. "
                    "Re-record with a larger --step to shrink it.", size)
    return OUTPUT


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the corridor visualiser page.")
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.scenario, args.seed)


if __name__ == "__main__":
    main()
