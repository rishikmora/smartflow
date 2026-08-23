"""Week 6: stress-test across demand scenarios and write the internal benchmark report.

Week 6's Definition of Done is "a viva-defensible benchmark report, including one
honest case where RL doesn't win". This script therefore does two things the earlier
weekly reports do not:

1. Compares every controller across all four demand scenarios (base / light / peak /
   asymmetric), so generalisation beyond the training demand is visible.
2. **Searches for losses rather than wins.** Every (scenario, metric) cell where the
   Week 5 policy is beaten by a baseline is enumerated automatically. The report is
   generated from the data, so it cannot quietly omit an unflattering column.

Produces:
    outputs/week6_scenarios.png       per-scenario avg-wait comparison
    outputs/week6_robustness.png      degradation from base demand
    outputs/week6_ablations.png       reward/architecture ablations on base demand
    BENCHMARK_REPORT.md               the internal benchmark report

Usage:
    python src/week6_benchmark.py
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import METRIC_LABELS, aggregate, comparison_chart, improvement, load_rows, markdown_table
from eval_marl_corridor import MARL_CSV
from scenarios import SCENARIOS, describe
from week4_config import CORRIDOR_TLS_IDS, OUTPUTS_DIR, ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

REPORT_PATH = os.path.join(ROOT, "BENCHMARK_REPORT.md")
SCENARIOS_PNG = os.path.join(OUTPUTS_DIR, "week6_scenarios.png")
ROBUSTNESS_PNG = os.path.join(OUTPUTS_DIR, "week6_robustness.png")
ABLATIONS_PNG = os.path.join(OUTPUTS_DIR, "week6_ablations.png")

CORE_METRICS = ["avg_wait_time_s", "max_queue_len", "throughput_veh"]
FAIRNESS_METRICS = ["wait_p95_s", "worst_vehicle_wait_s"]

BASELINES = ["fixed", "actuated"]
RL_MAIN = "marl_shared_w5"
SCENARIO_ORDER = ["base", "light", "peak", "asymmetric"]
MAIN_ORDER = ["fixed", "actuated", "marl_independent", RL_MAIN]
ABLATION_ORDER = [
    "fixed",
    "actuated",
    "marl_independent",
    "marl_shared_w5nofair",
    "marl_shared_w5nocoord",
    "marl_shared_w5gnn",
    "marl_shared_w5strong",
    RL_MAIN,
]

ABLATION_DESCRIPTIONS = {
    "marl_independent": "Week 4 — independent policies, local reward only",
    "marl_shared_w5nofair": "shared policy + green wave, **no** fairness constraint",
    "marl_shared_w5nocoord": "shared policy + fairness, **no** green-wave term",
    "marl_shared_w5gnn": "shared policy + full shaping, GAT encoder instead of MLP",
    "marl_shared_w5strong": "as the headline run but with a 20x stronger fairness weight",
    RL_MAIN: "Week 5 headline — shared policy, green wave + Lagrangian fairness",
}


def per_scenario(rows: list[dict[str, str]]) -> dict[str, dict]:
    """Aggregate corridor-scope metrics for each demand scenario.

    Args:
        rows: metric rows.

    Returns:
        ``{scenario: aggregate}`` for scenarios that have data.
    """
    out = {}
    for scenario in SCENARIO_ORDER:
        summary = aggregate(
            rows, CORE_METRICS + ["total_co2_kg"] + FAIRNESS_METRICS,
            scope="corridor", scenario=scenario,
        )
        if summary:
            out[scenario] = summary
    return out


def find_losses(by_scenario: dict[str, dict]) -> list[dict[str, object]]:
    """Enumerate every case where the headline RL policy loses to a baseline.

    This is the mechanism behind the "one honest case where RL doesn't win" DoD: the
    losses are found by search over the data, not chosen by hand.

    Args:
        by_scenario: output of :func:`per_scenario`.

    Returns:
        Loss records sorted worst-first.
    """
    losses: list[dict[str, object]] = []
    for scenario, summary in by_scenario.items():
        if RL_MAIN not in summary:
            continue
        for baseline in BASELINES:
            if baseline not in summary:
                continue
            for metric in CORE_METRICS + FAIRNESS_METRICS:
                rl_stats = summary[RL_MAIN].get(metric)
                base_stats = summary[baseline].get(metric)
                if rl_stats is None or base_stats is None:
                    continue
                pct = improvement(float(base_stats["mean"]), float(rl_stats["mean"]), metric)
                if pct < 0:
                    losses.append({
                        "scenario": scenario,
                        "baseline": baseline,
                        "metric": metric,
                        "rl": float(rl_stats["mean"]),
                        "base": float(base_stats["mean"]),
                        "pct": pct,
                    })
    losses.sort(key=lambda item: float(item["pct"]))
    return losses


def build_scenario_chart(by_scenario: dict[str, dict]) -> None:
    """Plot average wait per controller, grouped by demand scenario."""
    scenarios = [s for s in SCENARIO_ORDER if s in by_scenario]
    if not scenarios:
        log.warning("No scenario data; skipping the scenario chart.")
        return
    controllers = [c for c in MAIN_ORDER if any(c in by_scenario[s] for s in scenarios)]
    if not controllers:
        log.warning("No known controllers present; skipping the scenario chart.")
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.6 * len(scenarios), 4.8), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    palette = plt.get_cmap("tab10")
    # describe() parses every scenario route file, so resolve it once, not per subplot.
    scenario_info = describe()
    for ax, scenario in zip(axes, scenarios):
        labels, values, errors, colours = [], [], [], []
        for index, controller in enumerate(controllers):
            stats = by_scenario[scenario].get(controller, {}).get("avg_wait_time_s")
            if stats is None:
                continue
            labels.append(controller)
            values.append(float(stats["mean"]))
            errors.append(float(stats["std"]))
            colours.append(palette(index % 10))
        ax.bar(labels, values, yerr=errors, capsize=4, color=colours)
        ax.set_title(f"{scenario}\n({scenario_info[scenario]['vehicles']} vehicles)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel(METRIC_LABELS["avg_wait_time_s"])
    fig.suptitle("Week 6 — corridor average wait across demand scenarios (3 seeds)")
    fig.tight_layout()
    fig.savefig(SCENARIOS_PNG, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", SCENARIOS_PNG)


def build_robustness_chart(by_scenario: dict[str, dict]) -> None:
    """Plot each controller's average wait relative to its own base-demand result."""
    if "base" not in by_scenario:
        log.warning("No base-demand data; skipping the robustness chart.")
        return
    scenarios = [s for s in SCENARIO_ORDER if s in by_scenario]
    controllers = [c for c in MAIN_ORDER if c in by_scenario["base"]]
    if not controllers:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for controller in controllers:
        base_stats = by_scenario["base"][controller].get("avg_wait_time_s")
        if base_stats is None:
            continue
        xs, ys = [], []
        for scenario in scenarios:
            stats = by_scenario[scenario].get(controller, {}).get("avg_wait_time_s")
            if stats is None:
                continue
            xs.append(scenario)
            ys.append(float(stats["mean"]) / max(float(base_stats["mean"]), 1e-6))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=controller)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Avg wait relative to that controller's base demand")
    ax.set_title("Week 6 — degradation under demand shift (1.0 = same as base)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROBUSTNESS_PNG, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", ROBUSTNESS_PNG)


def write_report(rows: list[dict[str, str]]) -> None:
    """Write ``BENCHMARK_REPORT.md``.

    Args:
        rows: metric rows from the merged MARL metrics CSV.
    """
    by_scenario = per_scenario(rows)
    losses = find_losses(by_scenario)
    scenario_info = describe()

    parts = ["# SmartFlow Internal Benchmark Report (Phase A-B)\n\n"]
    parts.append(
        "Covers Weeks 1-6: fixed-time and actuated baselines, single-agent PPO on one\n"
        "junction, independent multi-agent PPO across the corridor, and the parameter-shared\n"
        "policy with green-wave shaping and a Lagrangian fairness constraint. Every number is\n"
        "a 3-seed mean ± standard deviation, measured by the same per-second metric collector\n"
        "(`src/metrics.py`) for RL and baselines alike.\n\n"
    )

    parts.append("## Network and demand\n\n")
    parts.append(
        f"`data/corridor.net.xml` — a {len(CORRIDOR_TLS_IDS)}-junction signalised grid "
        "(4x4 minus corners). Episodes are 1800 simulated seconds; agents decide every 5 s\n"
        "with a 2 s yellow, 5 s minimum green and 90 s maximum green.\n\n"
    )
    parts.append("| Scenario | Vehicles | Description |\n|---|---|---|\n")
    for name in SCENARIO_ORDER:
        if name in scenario_info:
            info = scenario_info[name]
            parts.append(f"| `{name}` | {info['vehicles']} | {info['description']} |\n")
    parts.append(
        "\nAll policies were trained on `base` only. `light`, `peak` and `asymmetric` are\n"
        "held-out demand, so those columns measure generalisation, not fit.\n"
    )

    parts.append("\n## Results by scenario\n\n")
    for scenario in SCENARIO_ORDER:
        if scenario not in by_scenario:
            continue
        parts.append(f"### `{scenario}` — {scenario_info[scenario]['description']}\n\n")
        parts.append(markdown_table(
            by_scenario[scenario], CORE_METRICS + FAIRNESS_METRICS,
            controller_order=MAIN_ORDER, baseline="fixed",
        ))
        parts.append("\n")

    parts.append("## Ablations (base demand)\n\n")
    if "base" in by_scenario:
        parts.append(markdown_table(
            by_scenario["base"], CORE_METRICS + FAIRNESS_METRICS,
            controller_order=ABLATION_ORDER, baseline="fixed",
        ))
        parts.append("\n")
        for controller, description in ABLATION_DESCRIPTIONS.items():
            if controller in by_scenario["base"]:
                parts.append(f"- `{controller}` — {description}\n")
    else:
        parts.append("_No base-demand rows found._\n")

    parts.append("\n## Where RL does not win\n\n")
    if not losses:
        parts.append(
            "No (scenario, metric) cell was found where the Week 5 policy loses to a\n"
            "baseline. That is itself worth flagging as suspicious rather than celebrated —\n"
            "check that every scenario was actually evaluated before trusting it.\n"
        )
    else:
        parts.append(
            "These cells were found by enumerating every (scenario, baseline, metric)\n"
            "combination and keeping the ones the RL policy loses. The list is generated from\n"
            "the data, so it cannot silently omit an unflattering result.\n\n"
        )
        parts.append("| Scenario | Metric | RL | Baseline | Baseline value | Gap |\n|---|---|---|---|---|---|\n")
        for loss in losses:
            parts.append(
                f"| `{loss['scenario']}` | {METRIC_LABELS.get(str(loss['metric']), loss['metric'])} | "
                f"{loss['rl']:.2f} | `{loss['baseline']}` | {loss['base']:.2f} | "
                f"{loss['pct']:+.1f}% |\n"
            )
        worst = losses[0]
        parts.append(
            f"\n**The honest failure case.** The clearest loss is `{worst['scenario']}` demand on\n"
            f"{METRIC_LABELS.get(str(worst['metric']), worst['metric'])}, where the policy is "
            f"{abs(float(worst['pct'])):.1f}% worse than `{worst['baseline']}`.\n"
        )

    from policy_agreement import markdown_summary

    agreement = markdown_summary()
    if agreement:
        parts.append(agreement)

    parts.append("\n## Methodology checks\n\n")
    parts.append(
        "Two ways this comparison could have been quietly unfair were checked rather than\n"
        "assumed away:\n\n"
        "- **Same simulator settings for every controller.** sumo-rl starts SUMO with\n"
        "  `--time-to-teleport -1`, while SUMO's own default teleports a vehicle stuck for\n"
        "  300 s. Teleporting clears gridlock and would have inflated the baselines'\n"
        "  throughput relative to the RL runs. Measured on base demand, a fixed-time run\n"
        "  records **zero teleports** and byte-identical metrics under either setting — but\n"
        "  `peak` is designed to gridlock, which is where it would bite, so all baselines\n"
        "  now go through `smartflow_env.baseline_sumo_cmd` and inherit sumo-rl's options.\n"
        "- **Same sampling rate for every metric.** RL controllers act every 5 simulated\n"
        "  seconds, so sampling once per decision would divide every waiting time by five.\n"
        "  The collector is driven once per simulated second for baselines and RL alike\n"
        "  (`ControlledSumoEnvironment._sumo_step`).\n"
    )

    parts.append("\n## Honest limitations\n\n")
    parts.append(
        "- **Single network, single route file.** Every result comes from one 12-junction\n"
        "  synthetic grid with one randomly generated demand set. Nothing here demonstrates\n"
        "  transfer to a real road network; the OSM corridor remains deferred from Week 1.\n"
        "- **Three seeds.** Enough to show variance, not enough for confidence intervals.\n"
        "  The Week 1 fixed-time baseline already showed high seed variance near saturation.\n"
        "- **Policies were trained on `base` demand only.** The other scenarios test\n"
        "  generalisation, and a policy retrained per scenario would very likely do better.\n"
        "- **`peak` drives the corridor past capacity.** At 150% demand the fixed-time\n"
        "  baseline gridlocks outright, so differences there partly reflect which controller\n"
        "  degrades more gracefully rather than which controls traffic better.\n"
        "- **Simulation only.** No claim is made about real-world or real-time performance;\n"
        "  nothing in this project has been load-tested.\n"
    )

    parts.append("\n## Definition of Done\n\n")
    parts.append("> Viva-defensible benchmark report, including one honest case where RL doesn't win.\n\n")
    parts.append(f"- Scenarios evaluated: **{len(by_scenario)} / {len(SCENARIOS)}**\n")
    parts.append(f"- Cases where RL loses to a baseline, enumerated above: **{len(losses)}**\n")
    met = len(by_scenario) == len(SCENARIOS) and bool(losses)
    parts.append(f"\n**Verdict: {'MET' if met else 'NOT MET'}**\n")

    parts.append("\n## Artifacts\n\n")
    parts.append("- `outputs/marl_metrics.csv` — every corridor-wide evaluation row\n")
    parts.append("- `outputs/week3_corridor_metrics.csv` — Week 3 single-junction rows\n")
    parts.append("- `outputs/week6_scenarios.png`, `outputs/week6_robustness.png`, `outputs/week6_ablations.png`\n")
    parts.append("- `outputs/week{3,4,5}_report.md` — the per-week reports\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    log.info("Wrote %s (DoD %s)", REPORT_PATH, "MET" if met else "NOT MET")


def main() -> None:
    """CLI entry point."""
    rows = load_rows(MARL_CSV)
    by_scenario = per_scenario(rows)
    build_scenario_chart(by_scenario)
    build_robustness_chart(by_scenario)
    if "base" in by_scenario:
        try:
            comparison_chart(
                by_scenario["base"], CORE_METRICS + FAIRNESS_METRICS, ABLATIONS_PNG,
                title="Week 6 — reward and architecture ablations (base demand, 3 seeds)",
                controller_order=ABLATION_ORDER,
            )
        except ValueError as exc:
            log.warning("Skipping the ablation chart: %s", exc)
    write_report(rows)


if __name__ == "__main__":
    main()
