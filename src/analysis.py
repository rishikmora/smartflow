"""Shared aggregation and plotting helpers for SmartFlow benchmark reports.

Every week's comparison chart, results table and improvement figure goes through
this module so the numbers in the README, the reports and the charts cannot drift
apart. Aggregation is always over seeds and always reports the standard deviation
alongside the mean — a 3-seed mean with no spread hides exactly the variance that
made Week 1's fixed-time baseline unreliable.
"""

from __future__ import annotations

import csv
import logging
import os
import statistics
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

METRIC_LABELS = {
    "avg_wait_time_s": "Avg wait (s)",
    "max_queue_len": "Max queue (veh)",
    "throughput_veh": "Throughput (veh)",
    "total_co2_kg": "CO2 (kg)",
    "wait_p95_s": "95th pct wait (s)",
    "worst_vehicle_wait_s": "Worst wait (s)",
}

# Metrics where a smaller number is better. Used to compute improvement signs.
LOWER_IS_BETTER = {
    "avg_wait_time_s", "max_queue_len", "total_co2_kg",
    "wait_p95_s", "worst_vehicle_wait_s",
}


def load_rows(csv_path: str) -> list[dict[str, str]]:
    """Read a metrics CSV into a list of dicts.

    Args:
        csv_path: path to a metrics CSV written by one of the eval harnesses.

    Returns:
        The rows, in file order.

    Raises:
        FileNotFoundError: if the CSV does not exist.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Metrics CSV not found: {csv_path}. Run the matching eval harness first."
        )
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float | None:
    """Parse a CSV cell to float, returning ``None`` for blanks and bad values."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate(
    rows: Iterable[dict[str, str]],
    metrics: Sequence[str],
    *,
    scope: str | None = None,
    scenario: str | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate metrics per controller, over seeds.

    Args:
        rows: metric rows.
        metrics: metric column names to aggregate.
        scope: keep only rows with this ``scope`` (``"junction"`` / ``"corridor"``).
        scenario: keep only rows with this ``scenario``, when the CSV has that column.

    Returns:
        ``{controller: {metric: {"mean", "std", "n", "min", "max"}}}``.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if scope is not None and row.get("scope") != scope:
            continue
        if scenario is not None and row.get("scenario", "base") != scenario:
            continue
        controller = row["controller"]
        target = buckets.setdefault(controller, {metric: [] for metric in metrics})
        for metric in metrics:
            value = _to_float(row.get(metric))
            if value is not None:
                target[metric].append(value)

    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for controller, per_metric in buckets.items():
        summary: dict[str, dict[str, float | int]] = {}
        for metric, values in per_metric.items():
            if not values:
                continue
            summary[metric] = {
                "mean": round(statistics.fmean(values), 3),
                "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 3),
                "n": len(values),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
        if summary:
            out[controller] = summary
    return out


def improvement(baseline: float, candidate: float, metric: str) -> float:
    """Return the percentage improvement of ``candidate`` over ``baseline``.

    Sign is normalised so a positive number always means "better", whichever
    direction the metric runs in.

    Args:
        baseline: the reference value.
        candidate: the value being judged.
        metric: metric name, used to decide the direction.

    Returns:
        Percentage improvement; ``0.0`` when the baseline is zero.
    """
    if baseline == 0:
        return 0.0
    delta = (candidate - baseline) / abs(baseline) * 100.0
    return round(-delta if metric in LOWER_IS_BETTER else delta, 2)


def markdown_table(
    summary: dict[str, dict[str, dict[str, float | int]]],
    metrics: Sequence[str],
    *,
    controller_order: Sequence[str] | None = None,
    baseline: str | None = None,
) -> str:
    """Render an aggregated summary as a Markdown table.

    Args:
        summary: output of :func:`aggregate`.
        metrics: metric columns to include, in order.
        controller_order: explicit row order; unlisted controllers are appended.
        baseline: if given, append a "% vs <baseline>" column per metric.

    Returns:
        A Markdown table, newline-terminated.
    """
    controllers = list(controller_order or [])
    controllers += [c for c in sorted(summary) if c not in controllers]
    controllers = [c for c in controllers if c in summary]

    header = ["Controller"] + [METRIC_LABELS.get(m, m) for m in metrics]
    if baseline and baseline in summary:
        header += [f"{METRIC_LABELS.get(m, m)} vs {baseline}" for m in metrics]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    for controller in controllers:
        cells = [controller]
        for metric in metrics:
            stats = summary[controller].get(metric)
            cells.append("n/a" if stats is None else f"{stats['mean']:.2f} ± {stats['std']:.2f}")
        if baseline and baseline in summary:
            for metric in metrics:
                stats = summary[controller].get(metric)
                base_stats = summary[baseline].get(metric)
                if controller == baseline:
                    cells.append("—")  # the reference row, not a 0% improvement
                elif stats is None or base_stats is None:
                    cells.append("n/a")
                else:
                    pct = improvement(float(base_stats["mean"]), float(stats["mean"]), metric)
                    cells.append(f"{pct:+.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def comparison_chart(
    summary: dict[str, dict[str, dict[str, float | int]]],
    metrics: Sequence[str],
    out_path: str,
    *,
    title: str,
    controller_order: Sequence[str] | None = None,
) -> str:
    """Draw a grouped bar chart, one subplot per metric, with std error bars.

    Args:
        summary: output of :func:`aggregate`.
        metrics: metrics to plot, one subplot each.
        out_path: destination PNG.
        title: figure title.
        controller_order: explicit bar order.

    Returns:
        ``out_path``.
    """
    controllers = list(controller_order or [])
    controllers += [c for c in sorted(summary) if c not in controllers]
    controllers = [c for c in controllers if c in summary]
    if not controllers:
        raise ValueError("Nothing to plot: no controllers present in the summary.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.6))
    if len(metrics) == 1:
        axes = [axes]

    palette = plt.get_cmap("tab10")
    for ax, metric in zip(axes, metrics):
        values, errors, labels, colours = [], [], [], []
        for index, controller in enumerate(controllers):
            stats = summary[controller].get(metric)
            if stats is None:
                continue
            values.append(float(stats["mean"]))
            errors.append(float(stats["std"]))
            labels.append(controller)
            colours.append(palette(index % 10))
        bars = ax.bar(labels, values, yerr=errors, capsize=4, color=colours)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def reward_curve_chart(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    out_path: str,
    *,
    title: str,
    xlabel: str = "Timesteps",
    ylabel: str = "Episode return",
) -> str:
    """Plot one or more training curves on shared axes.

    Args:
        series: ``{label: (x values, y values)}``.
        out_path: destination PNG.
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.

    Returns:
        ``out_path``.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (xs, ys) in sorted(series.items()):
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, linewidth=1.6, label=label, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def smooth(values: Sequence[float], window: int = 10) -> list[float]:
    """Return a trailing moving average, for readable training curves.

    Args:
        values: raw series.
        window: window length in samples.

    Returns:
        A list the same length as ``values``.
    """
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    running: list[float] = []
    for value in values:
        running.append(value)
        if len(running) > window:
            running.pop(0)
        out.append(statistics.fmean(running))
    return out


def read_monitor_csv(path: str) -> tuple[list[float], list[float]]:
    """Read a Stable-Baselines3 ``Monitor`` CSV into cumulative-steps / return series.

    Args:
        path: path to a ``*.monitor.csv`` file.

    Returns:
        ``(cumulative timesteps, episode returns)``; empty lists if the file is absent.
    """
    if not os.path.isfile(path):
        log.warning("Monitor file not found: %s", path)
        return [], []
    steps: list[float] = []
    returns: list[float] = []
    total = 0.0
    with open(path, newline="", encoding="utf-8") as handle:
        # first line is a JSON header comment written by Monitor
        first = handle.readline()
        if not first.startswith("#"):
            handle.seek(0)
        for row in csv.DictReader(handle):
            try:
                total += float(row["l"])
                returns.append(float(row["r"]))
                steps.append(total)
            except (KeyError, TypeError, ValueError):
                continue
    return steps, returns
