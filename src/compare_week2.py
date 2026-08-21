"""
Week 2 benchmark comparison chart.

Reads outputs/week2_benchmark_metrics.csv and produces a 3-panel bar chart
comparing fixed-time vs PPO controller across avg_wait_time_s, max_queue_len,
and throughput_veh.  Saves to outputs/week2_benchmark_comparison.png.

Usage:
    python src/compare_week2.py

Requires:
    Both 'fixed' and 'ppo' rows present in outputs/week2_benchmark_metrics.csv
    (populated by eval_benchmark.py on Days 3 and 5).
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from week2_config import BENCHMARK_CSV, BENCHMARK_PNG, OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

METRICS = [
    ("avg_wait_time_s", "Avg Wait Time (s)", "lower is better"),
    ("max_queue_len",   "Peak Queue Length (vehicles)", "lower is better"),
    ("throughput_veh",  "Throughput (vehicles)", "higher is better"),
]

COLOURS = {"fixed": "#4C72B0", "ppo": "#55A868"}
CONTROLLER_LABELS = {"fixed": "Fixed-Time", "ppo": "PPO (trained)"}


def _load_and_validate(csv_path: str) -> pd.DataFrame:
    """Read BENCHMARK_CSV and verify both controllers are present.

    Args:
        csv_path: path to the metrics CSV.

    Returns:
        DataFrame with all rows.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required controllers are missing.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Metrics CSV not found: {csv_path}\n"
            "Run eval_benchmark.py --controller fixed and --controller ppo first."
        )

    df = pd.read_csv(csv_path)
    present = set(df["controller"].unique())
    required = {"fixed", "ppo"}
    missing = required - present
    if missing:
        raise ValueError(
            f"Missing controllers in CSV: {missing}. "
            "Run eval_benchmark.py for the missing controller(s)."
        )
    return df


def _summarise(df: pd.DataFrame, metric: str) -> dict[str, tuple[float, float]]:
    """Compute mean and std for each controller on one metric.

    Args:
        df: full metrics DataFrame.
        metric: column name to summarise.

    Returns:
        Dict mapping controller name -> (mean, std).
    """
    return {
        ctrl: (grp[metric].mean(), grp[metric].std())
        for ctrl, grp in df.groupby("controller")
    }


def build_chart(df: pd.DataFrame, out_path: str) -> None:
    """Render and save the 3-panel comparison chart.

    Args:
        df: metrics DataFrame (all controllers, all seeds).
        out_path: destination path for the PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Week 2 Benchmark: Fixed-Time vs PPO\n"
        "2-way Single Intersection · 3 seeds",
        fontsize=13, fontweight="bold", y=1.02,
    )

    controllers = ["fixed", "ppo"]
    x_pos = np.arange(len(controllers))
    bar_width = 0.5

    for ax, (col, ylabel, direction) in zip(axes, METRICS):
        stats = _summarise(df, col)
        means = [stats[c][0] for c in controllers]
        stds  = [stats[c][1] for c in controllers]
        colors = [COLOURS[c] for c in controllers]
        labels = [CONTROLLER_LABELS[c] for c in controllers]

        bars = ax.bar(
            x_pos, means, bar_width,
            yerr=stds, capsize=6,
            color=colors,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
        )

        # annotate each bar with its mean ± std
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + (max(means) * 0.01),
                f"{m:.1f}±{s:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(f"{ylabel}\n({direction})", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Chart saved -> %s", out_path)


def _print_summary_table(df: pd.DataFrame) -> None:
    """Log a human-readable improvement summary to the console.

    Args:
        df: metrics DataFrame.
    """
    log.info("=" * 60)
    log.info("%-30s  %10s  %10s  %8s", "Metric", "Fixed", "PPO", "% change")
    log.info("-" * 60)

    for col, label, direction in METRICS:
        fixed_mean = df[df["controller"] == "fixed"][col].mean()
        ppo_mean   = df[df["controller"] == "ppo"][col].mean()
        pct = (ppo_mean - fixed_mean) / fixed_mean * 100
        sign = "+" if pct > 0 else ""
        log.info("%-30s  %10.1f  %10.1f  %s%.1f%%", label, fixed_mean, ppo_mean, sign, pct)

    log.info("=" * 60)


def main() -> None:
    """Load metrics, print table, save chart."""
    log.info("Reading metrics from %s ...", BENCHMARK_CSV)
    df = _load_and_validate(BENCHMARK_CSV)
    _print_summary_table(df)
    build_chart(df, BENCHMARK_PNG)


if __name__ == "__main__":
    main()
