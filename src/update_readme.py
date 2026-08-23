"""Regenerate the README's results block from the metrics CSVs.

The README quotes headline numbers for Weeks 1-6. Retyping those by hand is how a
README drifts away from the data it claims to summarise, so everything between the
``RESULTS:BEGIN`` / ``RESULTS:END`` markers is generated from the CSVs instead.

Prose outside the markers is hand-written and left untouched.

Usage:
    python src/update_readme.py
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import aggregate, improvement, load_rows, markdown_table
from week4_config import OUTPUTS_DIR, ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

README = os.path.join(ROOT, "README.md")
BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"

WEEK1_CSV = os.path.join(OUTPUTS_DIR, "metrics.csv")
WEEK2_CSV = os.path.join(OUTPUTS_DIR, "week2_benchmark_metrics.csv")
WEEK3_CSV = os.path.join(OUTPUTS_DIR, "week3_corridor_metrics.csv")
MARL_CSV = os.path.join(OUTPUTS_DIR, "marl_metrics.csv")
ONLINE_GLOB = os.path.join(OUTPUTS_DIR, "week5_online_learning_*.json")

CORE = ["avg_wait_time_s", "max_queue_len", "throughput_veh"]


def _safe_rows(path: str) -> list[dict[str, str]]:
    """Load a metrics CSV, returning an empty list if it is missing."""
    try:
        return load_rows(path)
    except FileNotFoundError:
        log.warning("Missing metrics file (section will be skipped): %s", path)
        return []


def _week1() -> str:
    """Render the Week 1 baseline section."""
    rows = _safe_rows(WEEK1_CSV)
    if not rows:
        return ""
    summary = aggregate(rows, CORE + ["total_co2_kg"])
    out = ["### Week 1 — fixed-time vs actuated baselines (corridor, 3 seeds)\n\n"]
    out.append(markdown_table(summary, CORE + ["total_co2_kg"],
                              controller_order=["fixed", "actuated"], baseline="fixed"))
    return "".join(out) + "\n"


def _week2() -> str:
    """Render the Week 2 benchmark section."""
    rows = _safe_rows(WEEK2_CSV)
    if not rows:
        return ""
    summary = aggregate(rows, CORE)
    out = ["### Week 2 — single-agent PPO on sumo-rl's standard benchmark (3 seeds)\n\n"]
    out.append(markdown_table(summary, CORE, controller_order=["fixed", "ppo"], baseline="fixed"))
    out.append("\nSanity check against published behaviour: `outputs/week2_literature_note.md`.\n")
    return "".join(out) + "\n"


def _week3() -> str:
    """Render the Week 3 single-junction section."""
    rows = _safe_rows(WEEK3_CSV)
    if not rows:
        return ""
    order = ["fixed", "actuated", "actuated_single", "ppo"]
    junction = aggregate(rows, CORE, scope="junction")
    corridor = aggregate(rows, CORE, scope="corridor")
    out = ["### Week 3 — PPO on one corridor junction (B1), other 11 fixed-time (3 seeds)\n\n"]
    out.append("**At the controlled junction** (the primary claim):\n\n")
    out.append(markdown_table(junction, CORE, controller_order=order, baseline="fixed"))
    out.append(
        "\n`actuated_single` is the controlled comparison — only B1 upgraded, same as `ppo`.\n\n"
        "**Corridor-wide** (one junction of twelve cannot move this much, which is the\n"
        "point of Week 4):\n\n"
    )
    out.append(markdown_table(corridor, CORE, controller_order=order, baseline="fixed"))
    return "".join(out) + "\n"


def _week456() -> str:
    """Render the corridor-wide multi-agent sections."""
    rows = _safe_rows(MARL_CSV)
    if not rows:
        return ""
    order = ["fixed", "actuated", "marl_independent", "marl_shared_w5"]
    base = aggregate(rows, CORE + ["wait_p95_s", "worst_vehicle_wait_s"],
                     scope="corridor", scenario="base")
    out = ["### Weeks 4-5 — corridor-wide multi-agent RL, base demand (3 seeds)\n\n"]
    out.append(markdown_table(base, CORE + ["wait_p95_s", "worst_vehicle_wait_s"],
                              controller_order=order, baseline="fixed"))
    out.append(
        "\n- `marl_independent` — Week 4: one policy per junction, local reward only.\n"
        "- `marl_shared_w5` — Week 5: one parameter-shared policy, green-wave shaping,\n"
        "  Lagrangian fairness constraint.\n"
    )

    # fairness ablation
    if "marl_shared_w5" in base and "marl_shared_w5nofair" in base:
        with_stats = base["marl_shared_w5"].get("worst_vehicle_wait_s")
        without_stats = base["marl_shared_w5nofair"].get("worst_vehicle_wait_s")
        if with_stats and without_stats:
            pct = improvement(float(without_stats["mean"]), float(with_stats["mean"]),
                              "worst_vehicle_wait_s")
            out.append(
                f"\n**Fairness constraint, measured by ablation:** worst-case vehicle wait is\n"
                f"{without_stats['mean']:.0f} s without the Lagrangian term and "
                f"{with_stats['mean']:.0f} s with it ({pct:+.1f}%).\n"
            )

    # scenario sweep
    scenario_rows = []
    for scenario in ("base", "light", "peak", "asymmetric"):
        summary = aggregate(rows, ["avg_wait_time_s"], scope="corridor", scenario=scenario)
        if not summary:
            continue
        cells = [f"`{scenario}`"]
        for controller in order:
            stats = summary.get(controller, {}).get("avg_wait_time_s")
            cells.append("n/a" if stats is None else f"{stats['mean']:.1f}")
        scenario_rows.append("| " + " | ".join(cells) + " |")
    if scenario_rows:
        out.append("\n### Week 6 — average wait (s) across demand scenarios\n\n")
        out.append("| Scenario | " + " | ".join(order) + " |\n")
        out.append("|" + "|".join(["---"] * (len(order) + 1)) + "|\n")
        out.append("\n".join(scenario_rows) + "\n")
        out.append("\nPolicies were trained on `base` only, so the other columns measure\n"
                   "generalisation. The full analysis, including every case where RL loses to a\n"
                   "baseline, is in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).\n")

    online_lines: list[str] = []
    for path in sorted(glob.glob(ONLINE_GLOB)):
        try:
            with open(path, encoding="utf-8") as handle:
                online = json.load(handle)
            online_lines.append(
                f"- deployed into `{online['scenario']}`: "
                f"{online['frozen_avg_wait_s']:.1f} s frozen -> "
                f"{online['final_avg_wait_s']:.1f} s after {online['rounds']} online rounds "
                f"({online['improvement_pct_final']:+.1f}%)\n"
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            log.warning("Could not summarise %s: %s", path, exc)
    if online_lines:
        out.append("\n### Week 5 — online learning\n\n")
        out.extend(online_lines)

    return "".join(out) + "\n"


def build_block() -> str:
    """Assemble the generated results block.

    Returns:
        Markdown to place between the README markers.
    """
    parts = [
        BEGIN,
        "\n<!-- Generated by src/update_readme.py — do not edit by hand. -->\n\n",
        "All figures are mean ± standard deviation over 3 seeds, measured by the same\n"
        "per-second collector (`src/metrics.py`) for RL and baselines alike.\n\n",
    ]
    for section in (_week1(), _week2(), _week3(), _week456()):
        if section:
            parts.append(section)
    parts.append(END)
    return "".join(parts)


def update() -> None:
    """Rewrite the README's results block in place.

    Raises:
        FileNotFoundError: if the README is missing.
        ValueError: if the markers are absent or out of order.
    """
    if not os.path.isfile(README):
        raise FileNotFoundError(f"README not found: {README}")
    with open(README, encoding="utf-8") as handle:
        content = handle.read()

    start = content.find(BEGIN)
    end = content.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"README must contain '{BEGIN}' and '{END}' markers, in that order, "
            "around the generated results block."
        )

    updated = content[:start] + build_block() + content[end + len(END):]
    with open(README, "w", encoding="utf-8") as handle:
        handle.write(updated)
    log.info("Updated the results block in %s", README)


def main() -> None:
    """CLI entry point."""
    update()


if __name__ == "__main__":
    main()
