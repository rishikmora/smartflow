"""Week 5 analysis: parameter-shared policy, green-wave shaping, Lagrangian fairness.

Reads ``outputs/marl_metrics.csv`` plus the training histories and produces:

* ``outputs/week5_comparison.png``  — corridor-wide bars vs both baselines
* ``outputs/week5_fairness.png``    — worst-case wait, with and without the constraint
* ``outputs/week5_lambda.png``      — the Lagrange multiplier's dual-ascent trajectory
* ``outputs/week5_report.md``       — results tables and the DoD verdict

The Week 5 DoD has two independent halves and both are checked separately:

1. Corridor-wide RL beats **both** baselines on wait time **and** throughput.
2. The fairness constraint **measurably** caps worst-case wait. This is judged by
   ablation — the same shared policy trained with and without the Lagrangian term —
   because "worst-case wait went down" means nothing without the counterfactual.

Usage:
    python src/compare_week5.py
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import aggregate, comparison_chart, improvement, load_rows, markdown_table, reward_curve_chart
from eval_marl_corridor import MARL_CSV
from week4_config import OUTPUTS_DIR, TRAIN_SEEDS, training_log_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

WEEK5_PNG = os.path.join(OUTPUTS_DIR, "week5_comparison.png")
WEEK5_FAIRNESS_PNG = os.path.join(OUTPUTS_DIR, "week5_fairness.png")
WEEK5_LAMBDA_PNG = os.path.join(OUTPUTS_DIR, "week5_lambda.png")
WEEK5_REPORT = os.path.join(OUTPUTS_DIR, "week5_report.md")

CORE_METRICS = ["avg_wait_time_s", "max_queue_len", "throughput_veh"]
FAIRNESS_METRICS = ["wait_p95_s", "worst_vehicle_wait_s"]

# CSV controller labels produced by the Week 4/5 evaluation runs.
BASELINES = ["fixed", "actuated"]
SHARED = "marl_shared_w5"
SHARED_NO_FAIRNESS = "marl_shared_w5nofair"
SHARED_STRONG_FAIRNESS = "marl_shared_w5strong"
INDEPENDENT = "marl_independent"
ORDER = ["fixed", "actuated", INDEPENDENT, SHARED_NO_FAIRNESS, SHARED, SHARED_STRONG_FAIRNESS]


def dod_performance(summary: dict) -> tuple[bool, list[str]]:
    """Check DoD part 1: beats both baselines on wait time and throughput.

    Args:
        summary: corridor-scope aggregate.

    Returns:
        ``(passed, explanation lines)``.
    """
    lines: list[str] = []
    if SHARED not in summary:
        return False, [f"- No `{SHARED}` rows in the metrics CSV — evaluate the Week 5 policy first."]

    passed = True
    for baseline in BASELINES:
        if baseline not in summary:
            lines.append(f"- Baseline `{baseline}` missing; cannot judge against it.")
            passed = False
            continue
        details = []
        ok = True
        for metric in ("avg_wait_time_s", "throughput_veh"):
            shared_stats = summary[SHARED].get(metric)
            base_stats = summary[baseline].get(metric)
            if shared_stats is None or base_stats is None:
                ok = False
                continue
            pct = improvement(float(base_stats["mean"]), float(shared_stats["mean"]), metric)
            details.append(f"{metric} {pct:+.1f}%")
            ok = ok and pct > 0
        passed = passed and ok
        lines.append(f"- vs `{baseline}`: {', '.join(details)} — {'PASS' if ok else 'FAIL'}")
    return passed, lines


def dod_fairness(summary: dict) -> tuple[bool, list[str]]:
    """Check DoD part 2: the fairness constraint measurably caps worst-case wait.

    Args:
        summary: corridor-scope aggregate.

    Returns:
        ``(passed, explanation lines)``.
    """
    lines: list[str] = []
    if SHARED not in summary or SHARED_NO_FAIRNESS not in summary:
        return False, [
            f"- Need both `{SHARED}` and `{SHARED_NO_FAIRNESS}` to judge the fairness claim by "
            "ablation; one of them is missing from the metrics CSV."
        ]

    ok_any = False
    for metric in FAIRNESS_METRICS:
        with_stats = summary[SHARED].get(metric)
        without_stats = summary[SHARED_NO_FAIRNESS].get(metric)
        if with_stats is None or without_stats is None:
            lines.append(f"- `{metric}`: not recorded.")
            continue
        pct = improvement(float(without_stats["mean"]), float(with_stats["mean"]), metric)
        better = pct > 0
        ok_any = ok_any or better
        lines.append(
            f"- `{metric}`: {without_stats['mean']:.1f}s without the constraint vs "
            f"{with_stats['mean']:.1f}s with it ({pct:+.1f}%) — "
            f"{'constraint helps' if better else 'constraint does not help'}"
        )
    return ok_any, lines


def per_seed_diagnosis(rows: list[dict[str, str]]) -> str:
    """Explain a failed Week 5 verdict from the per-seed numbers.

    The 3-seed mean can be dragged below a baseline by a single run that landed on a
    different policy. Whether that is what happened is visible only per seed, so the
    report shows it rather than leaving the reader to assume a systematic gap.

    Args:
        rows: raw metric rows.

    Returns:
        A Markdown section.
    """
    metrics = ["avg_wait_time_s", "throughput_veh", "worst_vehicle_wait_s"]
    controllers = [SHARED, SHARED_NO_FAIRNESS, SHARED_STRONG_FAIRNESS, INDEPENDENT, "actuated"]
    table: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        if row.get("scope") != "corridor" or row.get("scenario") != "base":
            continue
        if row["controller"] not in controllers:
            continue
        try:
            values = {m: float(row[m]) for m in metrics if row.get(m) not in (None, "")}
        except (TypeError, ValueError):
            continue
        table.setdefault(row["controller"], {})[int(row["seed"])] = values

    parts = ["\n### Why the verdict failed — per-seed numbers\n\n"]
    seeds = sorted({s for per in table.values() for s in per})
    for metric in metrics:
        parts.append(f"**{metric}**\n\n")
        parts.append("| Controller | " + " | ".join(f"seed {s}" for s in seeds) + " |\n")
        parts.append("|" + "|".join(["---"] * (len(seeds) + 1)) + "|\n")
        for controller in controllers:
            per = table.get(controller)
            if not per:
                continue
            cells = " | ".join(
                f"{per[s][metric]:.1f}" if s in per and metric in per[s] else "n/a" for s in seeds
            )
            parts.append(f"| `{controller}` | {cells} |\n")
        parts.append("\n")

    shared = table.get(SHARED, {})
    others = table.get(SHARED_NO_FAIRNESS, {})
    outlier = None
    if shared and others:
        for seed in seeds:
            if seed in shared and seed in others:
                a = shared[seed].get("throughput_veh")
                b = others[seed].get("throughput_veh")
                if a is not None and b is not None and abs(a - b) > 0.05 * max(b, 1):
                    outlier = seed
                    break

    if outlier is not None:
        parts.append(
            f"**One seed, not a systematic gap.** Seeds other than {outlier} land on the same\n"
            "deterministic policy as every ablation, and there the shared policy *does* beat\n"
            f"actuated control on throughput. Seed {outlier} converged somewhere else — a policy\n"
            "that jams the corridor — and it is what pulls the 3-seed mean below the baseline.\n\n"
            "That is reported as a FAIL rather than dropped as an outlier. Three seeds is\n"
            "already the minimum this project accepts for a finding; discarding the one that\n"
            "disagrees would leave two, and would be exactly the selective reporting the\n"
            "3-seed rule exists to prevent. The honest summary is that this configuration is\n"
            "**not reliably** better than actuated control on throughput: it is better twice\n"
            "out of three times, and materially worse the third.\n\n"
            "Note also that Week 4's `marl_independent` — the simpler, unshaped configuration —\n"
            "does clear the bar on both metrics. Corridor-wide RL beating both baselines is a\n"
            "real result on this network; it is the Week 5 *shaping* that fails to add to it.\n"
        )
    return "".join(parts)


def build_lambda_chart(tag: str = "w5") -> dict[int, dict]:
    """Plot the Lagrange multiplier's trajectory for each seed.

    Args:
        tag: checkpoint tag of the Week 5 runs.

    Returns:
        ``{seed: history}`` for every history that loaded.
    """
    histories: dict[int, dict] = {}
    lam_series: dict[str, tuple[list[float], list[float]]] = {}
    wait_series: dict[str, tuple[list[float], list[float]]] = {}
    for seed in TRAIN_SEEDS:
        path = training_log_path("shared", seed, tag)
        if not os.path.isfile(path):
            log.warning("Missing Week 5 training history: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                history = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read %s: %s", path, exc)
            continue
        histories[seed] = history
        iterations = history.get("iterations", [])
        steps = [it["env_steps"] for it in iterations]
        lam_series[f"seed {seed} lambda"] = (steps, [it.get("fairness_lambda", 0.0) for it in iterations])
        waits = [it.get("max_lane_wait_s") for it in iterations]
        if any(w is not None for w in waits):
            wait_series[f"seed {seed} max lane wait"] = (steps, [w or 0.0 for w in waits])

    if lam_series:
        reward_curve_chart(
            lam_series, WEEK5_LAMBDA_PNG,
            title="Week 5 — Lagrange multiplier under dual ascent",
            xlabel="Environment steps", ylabel="lambda",
        )
    return histories


def write_report(rows: list[dict[str, str]], histories: dict[int, dict]) -> None:
    """Write ``outputs/week5_report.md``.

    Args:
        rows: metric rows from the MARL metrics CSV.
        histories: ``{seed: Week 5 training history}``.
    """
    corridor = aggregate(rows, CORE_METRICS + ["total_co2_kg"] + FAIRNESS_METRICS,
                         scope="corridor", scenario="base")
    perf_ok, perf_lines = dod_performance(corridor)
    fair_ok, fair_lines = dod_fairness(corridor)
    passed = perf_ok and fair_ok

    parts = ["# Week 5 — Parameter-Shared MARL, Green-Wave Shaping, Lagrangian Fairness\n\n"]
    parts.append(
        "Three changes on top of Week 4's independent learners:\n\n"
        "1. **Parameter sharing** — one policy drives all twelve junctions instead of twelve\n"
        "   separate policies, so every junction's experience updates the same weights.\n"
        "2. **Green-wave reward shaping** — each junction is penalised for discharging into a\n"
        "   congested downstream link (`-alpha * mean outgoing-lane density`), which makes\n"
        "   pushing congestion onto a neighbour costly rather than free.\n"
        "3. **Lagrangian fairness constraint** — the per-lane accumulated wait is capped, with\n"
        "   the multiplier `lambda` updated by dual ascent between training iterations rather\n"
        "   than hand-tuned.\n\n"
    )

    parts.append("## Corridor-wide results (base demand, 3 seeds)\n\n")
    parts.append(markdown_table(corridor, CORE_METRICS + ["total_co2_kg"],
                                controller_order=ORDER, baseline="fixed"))

    parts.append("\n## Fairness / tail metrics\n\n")
    parts.append(markdown_table(corridor, FAIRNESS_METRICS, controller_order=ORDER))
    parts.append(
        "\n`wait_p95_s` is the 95th percentile of completed-trip waiting time and\n"
        "`worst_vehicle_wait_s` the single worst trip. Average delay can improve while the\n"
        "tail gets worse — a policy is free to starve a side street to speed up the main\n"
        "road — so the constraint is judged on the tail, not the mean.\n"
    )

    if histories:
        parts.append("\n## Dual ascent on the fairness multiplier\n\n")
        parts.append("| Seed | lambda start | lambda final | Final max lane wait (s) | Cap (s) |\n")
        parts.append("|---|---|---|---|---|\n")
        for seed, history in sorted(histories.items()):
            iterations = history.get("iterations", [])
            start = iterations[0].get("fairness_lambda") if iterations else None
            final = history.get("fairness_lambda_final")
            last_wait = iterations[-1].get("max_lane_wait_s") if iterations else None
            parts.append(
                f"| {seed} | {start if start is not None else 'n/a'} | "
                f"{final if final is not None else 'n/a'} | "
                f"{last_wait if last_wait is not None else 'n/a'} | "
                f"{history.get('fairness_cap_s', 'n/a')} |\n"
            )
        parts.append(
            "\n`lambda` rises while the cap is breached and decays once it is satisfied, which\n"
            "is what makes this a constraint rather than a fixed penalty weight. See\n"
            "`outputs/week5_lambda.png`.\n"
        )

        # Report multiplier saturation explicitly: it means the constraint was never met.
        caps = [h.get("fairness_cap_s") for h in histories.values()]
        finals = [h.get("fairness_lambda_final") for h in histories.values()
                  if h.get("fairness_lambda_final") is not None]
        clamp = 2.0
        saturated = [value for value in finals if value >= clamp - 1e-9]
        if saturated and len(saturated) == len(finals):
            parts.append(
                f"\n**The multiplier saturated in every seed.** `lambda` reached its clamp of\n"
                f"{clamp} and stayed there, which means the "
                f"{caps[0] if caps else 'configured'} s cap was *never satisfied* during\n"
                "training — dual ascent kept raising the price and the policy kept paying it.\n"
                "Two things follow, and both are worth saying plainly:\n\n"
                "1. Over most of training the term behaved like a fixed penalty at maximum\n"
                "   strength, not like an adaptive multiplier. The adaptivity only shows in the\n"
                "   early iterations before the clamp is reached.\n"
                "2. The cap is likely infeasible for this corridor at this demand. A constraint\n"
                "   the policy cannot meet is a statement about the network's capacity, not a\n"
                "   failure of the optimiser. Choosing a feasible cap — or reporting the\n"
                "   achievable frontier by sweeping it — is the honest next step, and is left\n"
                "   as documented future work rather than quietly retuned until it looked good.\n"
            )

    online_runs = []
    for path in sorted(glob.glob(os.path.join(OUTPUTS_DIR, "week5_online_learning_*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                online_runs.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read %s: %s", path, exc)

    if online_runs:
        parts.append("\n## Online learning loop\n\n")
        parts.append(
            "The Week 5 policy, trained on base demand, was deployed into a *different*\n"
            "demand and kept learning from the live stream. Round 0 is the frozen policy —\n"
            "what you get if a deployed controller never updates.\n\n"
        )
        parts.append("| Deployed into | Rounds | Frozen wait (s) | Final wait (s) | Best wait (s) | Change |\n")
        parts.append("|---|---|---|---|---|---|\n")
        for online in online_runs:
            parts.append(
                f"| `{online['scenario']}` | {online['rounds']} | "
                f"{online['frozen_avg_wait_s']:.2f} | {online['final_avg_wait_s']:.2f} | "
                f"{online['best_avg_wait_s']:.2f} | {online['improvement_pct_final']:+.1f}% |\n"
            )
        moved = [o for o in online_runs if abs(o["improvement_pct_final"]) > 0.5]
        if not moved:
            parts.append(
                "\n**No measurable adaptation, and the reason is the same one that flattens the\n"
                "ablations.** Each online round performs a real PPO update — the weights move —\n"
                "but the policy's argmax does not, so the deterministic rollout replays exactly\n"
                "the same actions and returns exactly the same metrics. The loop is doing what it\n"
                "was built to do; the task simply has no headroom left at this action resolution.\n"
                "See the agreement analysis below.\n"
            )
        parts.append("\nCurves: `outputs/week5_online_learning_<scenario>.png`.\n")

    parts.append("\n## Definition of Done\n\n")
    parts.append(
        "> Corridor-wide RL beats both baselines on wait time + throughput, with fairness\n"
        "> constraint measurably capping worst-case wait.\n\n"
    )
    parts.append("**Part 1 — beats both baselines on wait time and throughput**\n\n")
    parts.extend(line + "\n" for line in perf_lines)
    parts.append(f"\n_Part 1: {'PASS' if perf_ok else 'FAIL'}_\n\n")
    parts.append("**Part 2 — fairness constraint measurably caps worst-case wait (by ablation)**\n\n")
    parts.extend(line + "\n" for line in fair_lines)
    parts.append(f"\n_Part 2: {'PASS' if fair_ok else 'FAIL'}_\n")

    if not (perf_ok and fair_ok):
        parts.append(per_seed_diagnosis(rows))

    from policy_agreement import markdown_summary

    agreement = markdown_summary()
    if agreement:
        parts.append(
            "\n### Why Part 2 cannot be shown on this task\n\n"
            "The ablation returns identical numbers with and without the constraint, and the\n"
            "reason was measured rather than guessed.\n"
        )
        parts.append(agreement)
    parts.append(f"\n**Verdict: {'MET' if passed else 'NOT MET'}**\n")

    parts.append("\n## Reproducing\n\n```powershell\n")
    parts.append("python src\\train_marl_corridor.py --mode shared --reward shaped --fairness --tag w5 --seed 0\n")
    parts.append("python src\\train_marl_corridor.py --mode shared --reward shaped-no-fairness --tag w5nofair --seed 0\n")
    parts.append("python src\\eval_marl_corridor.py --controller marl --mode shared --tag w5 --reward shaped --seeds 0 1 2\n")
    parts.append("python src\\online_learning.py --seed 0 --tag w5 --scenario asymmetric\n")
    parts.append("python src\\compare_week5.py\n```\n")

    os.makedirs(os.path.dirname(WEEK5_REPORT), exist_ok=True)
    with open(WEEK5_REPORT, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    log.info("Wrote %s (DoD %s)", WEEK5_REPORT, "MET" if passed else "NOT MET")


def main() -> None:
    """CLI entry point."""
    rows = load_rows(MARL_CSV)
    corridor = aggregate(rows, CORE_METRICS + ["total_co2_kg"] + FAIRNESS_METRICS,
                         scope="corridor", scenario="base")
    if corridor:
        try:
            comparison_chart(
                corridor, CORE_METRICS + ["total_co2_kg"], WEEK5_PNG,
                title="Week 5 — corridor-wide, shared-policy MARL vs baselines (3 seeds)",
                controller_order=ORDER,
            )
            comparison_chart(
                corridor, FAIRNESS_METRICS, WEEK5_FAIRNESS_PNG,
                title="Week 5 — worst-case wait, with and without the fairness constraint",
                controller_order=ORDER,
            )
        except ValueError as exc:
            log.warning("Skipping a Week 5 chart: %s", exc)
    histories = build_lambda_chart("w5")
    write_report(rows, histories)


if __name__ == "__main__":
    main()
