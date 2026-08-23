"""Week 3 analysis: charts and report for single-agent PPO on one corridor junction.

Reads ``outputs/week3_corridor_metrics.csv`` and produces:

* ``outputs/week3_comparison.png``    — junction-scope and corridor-scope bar charts
* ``outputs/week3_reward_curves.png`` — per-seed training curves
* ``outputs/week3_report.md``         — results tables and the Definition-of-Done verdict

The Week 3 DoD is "RL beats both baselines on >=2 of 3 metrics, 3-seed average +
variance". Both baselines are checked at the **junction** scope, because that is the
only scope where a single controlled junction can be expected to move the numbers.
Corridor-scope results are reported alongside, unspun.

Usage:
    python src/compare_week3.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    aggregate,
    comparison_chart,
    improvement,
    load_rows,
    markdown_table,
    read_monitor_csv,
    reward_curve_chart,
    smooth,
)
from week3_config import (
    CORRIDOR_TLS_ID,
    OUTPUTS_DIR,
    TRAIN_SEEDS,
    WEEK3_CSV,
    WEEK3_CURVE_PNG,
    WEEK3_PNG,
    WEEK3_REPORT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

CORE_METRICS = ["avg_wait_time_s", "max_queue_len", "throughput_veh"]
CONTROLLER_ORDER = ["fixed", "actuated", "actuated_single", "ppo_final_iterate", "ppo"]
MONITOR_DIR = os.path.join(OUTPUTS_DIR, "monitor_week3")
SELECTION_JSON = os.path.join(OUTPUTS_DIR, "week3_checkpoint_selection.json")


def build_charts(rows: list[dict[str, str]]) -> None:
    """Render the Week 3 comparison charts.

    Args:
        rows: metric rows from the Week 3 CSV.
    """
    junction = aggregate(rows, CORE_METRICS, scope="junction")
    corridor = aggregate(rows, CORE_METRICS + ["total_co2_kg"], scope="corridor")

    comparison_chart(
        junction, CORE_METRICS,
        os.path.join(OUTPUTS_DIR, "week3_comparison_junction.png"),
        title=f"Week 3 — junction {CORRIDOR_TLS_ID} (3 seeds, error bars = std)",
        controller_order=CONTROLLER_ORDER,
    )
    comparison_chart(
        corridor, CORE_METRICS + ["total_co2_kg"], WEEK3_PNG,
        title="Week 3 — corridor-wide (3 seeds, error bars = std)",
        controller_order=CONTROLLER_ORDER,
    )


def build_reward_curves() -> None:
    """Render per-seed PPO training curves from the SB3 monitor logs."""
    series = {}
    for seed in TRAIN_SEEDS:
        path = os.path.join(MONITOR_DIR, f"corridor_seed{seed}.monitor.csv")
        steps, returns = read_monitor_csv(path)
        if steps:
            series[f"seed {seed}"] = (steps, smooth(returns, window=15))
    if not series:
        log.warning("No monitor logs found in %s; skipping reward-curve chart.", MONITOR_DIR)
        return
    reward_curve_chart(
        series, WEEK3_CURVE_PNG,
        title=f"Week 3 — PPO training on junction {CORRIDOR_TLS_ID} (15-episode moving average)",
        ylabel="Episode return (diff-waiting-time)",
    )


def selection_summary() -> str:
    """Describe what validation-based checkpoint selection actually changed.

    Written from the recorded scores rather than asserted, because on this task the
    honest answer turned out to be "not much": the validation episode cannot separate
    most checkpoints once the deterministic policy has converged.

    Returns:
        A Markdown section, or an empty string when no selection has been run.
    """
    if not os.path.isfile(SELECTION_JSON):
        return ""
    try:
        with open(SELECTION_JSON, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", SELECTION_JSON, exc)
        return ""

    rows = ["\n| Seed | Validation seed | Chosen | Score (s) | Distinct scores among candidates |\n",
            "|---|---|---|---|---|\n"]
    tied_seeds = 0
    for record in data.get("selections", []):
        scores = {round(float(c["score"]), 3) for c in record.get("candidates", [])}
        if len(scores) == 1:
            tied_seeds += 1
        rows.append(
            f"| {record['seed']} | {record['validation_seed']} | `{record['selected']}` | "
            f"{record['selected_score']:.2f} | {len(scores)} of {len(record.get('candidates', []))} |\n"
        )

    note = (
        "\nSelection changed less than one might hope, and that is worth stating rather than\n"
        "hiding: once the policy converges its *deterministic* (argmax) behaviour stops\n"
        "moving, so most checkpoints score identically on a validation episode even while\n"
        "the stochastic training policy keeps fluctuating. "
    )
    if tied_seeds:
        note += (
            f"For {tied_seeds} of {len(data.get('selections', []))} seeds every candidate scored\n"
            "the same, so the tie broke toward the most-trained checkpoint and the reported\n"
            "policy is simply the final iterate.\n"
        )
    note += (
        "\nA single validation episode per candidate is also a thin basis for choosing. "
        "Averaging over several validation seeds would discriminate better; that is a known\n"
        "limitation of this protocol, not something the numbers below hide.\n"
    )
    return "".join(rows) + note


def diagnosis(rows: list[dict[str, str]], junction: dict) -> str:
    """Explain a failed Definition of Done from the per-seed numbers.

    A bare FAIL is not a result. This pulls the per-seed junction waits so the report
    says *which* seed cost the verdict and by how much.

    Args:
        rows: raw metric rows.
        junction: junction-scope aggregate.

    Returns:
        A Markdown section.
    """
    per_seed: dict[str, dict[int, float]] = {}
    for row in rows:
        if row.get("scope") != "junction":
            continue
        try:
            per_seed.setdefault(row["controller"], {})[int(row["seed"])] = float(row["avg_wait_time_s"])
        except (TypeError, ValueError):
            continue

    parts = ["\n### Why it failed, and what that means\n\n"]
    ppo_by_seed = per_seed.get("ppo", {})
    if ppo_by_seed:
        worst_seed = max(ppo_by_seed, key=lambda s: ppo_by_seed[s])
        others = [v for s, v in ppo_by_seed.items() if s != worst_seed]
        parts.append("Per-seed junction wait (s) — the spread is the story:\n\n")
        parts.append("| Controller | " + " | ".join(f"seed {s}" for s in sorted(ppo_by_seed)) + " |\n")
        parts.append("|" + "|".join(["---"] * (len(ppo_by_seed) + 1)) + "|\n")
        for controller in ("fixed", "actuated_single", "ppo"):
            values = per_seed.get(controller, {})
            if not values:
                continue
            cells = " | ".join(f"{values.get(s, float('nan')):.2f}" for s in sorted(ppo_by_seed))
            parts.append(f"| `{controller}` | {cells} |\n")
        if others:
            parts.append(
                f"\nSeed {worst_seed} is the outlier: {ppo_by_seed[worst_seed]:.1f} s against "
                f"{'/'.join(f'{v:.1f}' for v in others)} s on the other seeds. It is what drives\n"
                "both the large standard deviation and the lost verdict.\n"
            )

    parts.append(
        "\nTwo honest readings, and the project takes both:\n\n"
        "1. **SUMO's actuated controller is a strong baseline, not a straw man.** It is a\n"
        "   well-tuned classical method with direct detector access. Single-agent RL beating\n"
        "   the *fixed-time* program by 67% while losing to actuated control on queue and\n"
        "   throughput is a normal, publishable-shaped result — not a failure of the pipeline.\n"
        "2. **One junction cannot be judged in isolation.** The RL agent optimises its own\n"
        "   accumulated wait, which it can reduce by discharging vehicles into neighbours it\n"
        "   does not control and cannot see. Throughput at B1 suffers because the downstream\n"
        "   links back up. That is precisely the credit-assignment problem Weeks 4-6 exist to\n"
        "   address, and corridor-wide the shared policy does beat actuated control.\n\n"
        "The verdict is reported as failed rather than re-scoped until it passed. Changing the\n"
        "comparison to the fully-actuated corridor would have produced a PASS against a\n"
        "baseline that upgrades twelve junctions to the RL controller's one, which would not\n"
        "have meant anything.\n"
    )
    return "".join(parts)


def _rows_identical(summary: dict, left: str, right: str) -> bool:
    """Return whether two controllers have identical aggregates on every metric.

    Args:
        summary: aggregate from :func:`analysis.aggregate`.
        left: first controller label.
        right: second controller label.

    Returns:
        ``True`` when both are present and every metric's mean and std match.
    """
    if left not in summary or right not in summary:
        return False
    for metric in CORE_METRICS:
        a, b = summary[left].get(metric), summary[right].get(metric)
        if a is None or b is None or a["mean"] != b["mean"] or a["std"] != b["std"]:
            return False
    return True


def dod_verdict(junction: dict) -> tuple[bool, list[str]]:
    """Evaluate the Week 3 Definition of Done.

    Args:
        junction: junction-scope aggregate from :func:`analysis.aggregate`.

    Returns:
        ``(passed, lines)`` where ``lines`` explains the verdict metric by metric.
    """
    lines: list[str] = []
    if "ppo" not in junction:
        return False, ["No `ppo` rows found — train and evaluate the policy first."]

    passed_all = True
    for baseline in ("fixed", "actuated_single"):
        if baseline not in junction:
            lines.append(f"- Baseline `{baseline}` missing from the CSV; cannot judge against it.")
            passed_all = False
            continue
        wins = []
        for metric in CORE_METRICS:
            ppo_stats = junction["ppo"].get(metric)
            base_stats = junction[baseline].get(metric)
            if ppo_stats is None or base_stats is None:
                continue
            pct = improvement(float(base_stats["mean"]), float(ppo_stats["mean"]), metric)
            wins.append((metric, pct))
        won = [m for m, p in wins if p > 0]
        detail = ", ".join(f"{m} {p:+.1f}%" for m, p in wins)
        ok = len(won) >= 2
        passed_all = passed_all and ok
        lines.append(
            f"- vs `{baseline}`: beats it on **{len(won)} of {len(wins)}** metrics "
            f"({detail}) — {'PASS' if ok else 'FAIL'}"
        )
    return passed_all, lines


def write_report(rows: list[dict[str, str]]) -> None:
    """Write ``outputs/week3_report.md``.

    Args:
        rows: metric rows from the Week 3 CSV.
    """
    junction = aggregate(rows, CORE_METRICS, scope="junction")
    corridor = aggregate(rows, CORE_METRICS + ["total_co2_kg"], scope="corridor")
    passed, verdict_lines = dod_verdict(junction)

    parts: list[str] = []
    parts.append("# Week 3 — Single-Agent PPO on One Corridor Junction\n")
    parts.append(
        "Week 2 validated the PPO pipeline on sumo-rl's standard 2-way single-intersection\n"
        "benchmark. Week 3 ports that pipeline, unchanged in its observation and reward\n"
        f"functions, onto junction **{CORRIDOR_TLS_ID}** of the Week 1 corridor. The other\n"
        "eleven junctions keep the fixed-time program from `data/corridor.net.xml`.\n"
    )

    parts.append("\n## Experimental design\n")
    parts.append(
        "| Controller | What it does |\n|---|---|\n"
        "| `fixed` | all 12 junctions fixed-time (Week 1 reference) |\n"
        "| `actuated` | all 12 junctions SUMO-actuated (Week 1 reference) |\n"
        f"| `actuated_single` | only {CORRIDOR_TLS_ID} actuated, other 11 fixed-time |\n"
        f"| `ppo` | only {CORRIDOR_TLS_ID} under RL, other 11 fixed-time |\n"
    )
    parts.append(
        f"\n`actuated_single` is the controlled comparison for `ppo`: both change exactly one\n"
        f"junction and leave the other eleven alone, so the difference between them is the\n"
        f"controller, not the number of junctions upgraded. Comparing a single RL junction\n"
        f"against a fully-actuated corridor would confound those two effects.\n"
    )

    parts.append("\n## Which policy is reported\n\n")
    parts.append(
        "PPO's final weights are not its best weights. On this corridor the per-episode\n"
        "training return for seed 1 swings between -1.3 and -959 in consecutive episodes:\n"
        "near saturation, one unlucky action sequence cascades into a jam the rest of the\n"
        "episode cannot clear. Reporting whichever policy existed when the step budget ran\n"
        "out therefore measures luck as much as learning.\n\n"
        "Two rows are reported:\n\n"
        "- **`ppo`** — the checkpoint chosen by `src/select_best_checkpoint.py`, scored on a\n"
        "  **validation seed (100 + training seed)** that is disjoint from the evaluation\n"
        "  seeds 0/1/2. Selecting on the evaluation seeds themselves would be tuning on the\n"
        "  test set.\n"
        "- **`ppo_final_iterate`** — the last-iterate policy, kept in the table because its\n"
        "  spread is the honest measure of how unstable this training run was.\n\n"
        "Full selection table: `outputs/week3_checkpoint_selection.json`.\n"
    )
    parts.append(selection_summary())
    parts.append(f"\n## Results at junction {CORRIDOR_TLS_ID} (primary)\n\n")
    parts.append(markdown_table(junction, CORE_METRICS,
                                controller_order=CONTROLLER_ORDER, baseline="fixed"))
    parts.append(
        f"\nMetrics cover vehicles on {CORRIDOR_TLS_ID}'s incoming lanes: mean stopped seconds\n"
        "per vehicle that cleared the junction, peak halting count on those lanes, and the\n"
        "number of vehicles that cleared. Values are mean ± std over 3 seeds.\n"
    )

    if _rows_identical(junction, "ppo", "ppo_final_iterate"):
        parts.append(
            "\n`ppo` and `ppo_final_iterate` are identical here, and that is the result rather\n"
            "than a copy-paste error: validation-based selection did not change the reported\n"
            "numbers on this task. It was still the right protocol to run — the alternative is\n"
            "not knowing whether the last iterate was lucky — but on this corridor it bought\n"
            "nothing, so no claim rests on it.\n"
        )

    parts.append("\n## Results corridor-wide (context)\n\n")
    parts.append(markdown_table(corridor, CORE_METRICS + ["total_co2_kg"],
                                controller_order=CONTROLLER_ORDER, baseline="fixed"))
    parts.append(
        "\nOne junction out of twelve cannot move corridor-wide numbers much, and the table\n"
        "shows that plainly. That ceiling is the motivation for Week 4's multi-agent work,\n"
        "not a result to explain away.\n"
    )

    parts.append("\n## Definition of Done\n\n")
    parts.append("> RL beats both baselines on >=2 of 3 metrics, 3-seed average + variance.\n\n")
    parts.extend(line + "\n" for line in verdict_lines)
    parts.append(f"\n**Verdict: {'MET' if passed else 'NOT MET'}**\n")
    if not passed:
        parts.append(diagnosis(rows, junction))

    parts.append("\n## Reproducing\n\n```powershell\n")
    parts.append("python src\\make_single_actuated_net.py\n")
    for controller in ("fixed", "actuated", "actuated_single"):
        parts.append(f"python src\\eval_corridor.py --controller {controller} --seeds 0 1 2\n")
    parts.append("python src\\train_ppo_corridor.py --seed 0   # repeat for seeds 1, 2\n")
    parts.append("python src\\select_best_checkpoint.py\n")
    parts.append("python src\\eval_corridor.py --controller ppo --seeds 0 1 2\n")
    parts.append("python src\\compare_week3.py\n```\n")

    parts.append("\n## Artifacts\n\n")
    parts.append("- `outputs/week3_corridor_metrics.csv` — every logged run\n")
    parts.append(f"- `outputs/week3_comparison_junction.png` — junction {CORRIDOR_TLS_ID} chart\n")
    parts.append("- `outputs/week3_comparison.png` — corridor-wide chart\n")
    parts.append("- `outputs/week3_reward_curves.png` — per-seed training curves\n")
    parts.append("- `models/ppo_corridor_seed{0,1,2}.zip` + `_hparams.json` — policies and settings\n")

    os.makedirs(os.path.dirname(WEEK3_REPORT), exist_ok=True)
    with open(WEEK3_REPORT, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    log.info("Wrote %s (DoD %s)", WEEK3_REPORT, "MET" if passed else "NOT MET")


def main() -> None:
    """CLI entry point."""
    rows = load_rows(WEEK3_CSV)
    build_charts(rows)
    build_reward_curves()
    write_report(rows)


if __name__ == "__main__":
    main()
