"""Create the Week 3 fixed/actuated/PPO comparison chart and report."""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from week3_config import WEEK3_CSV, WEEK3_PNG, WEEK3_REPORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

METRICS = [
    ("avg_wait_time_s", "Avg Wait Time (s)", "lower is better"),
    ("max_queue_len", "Peak Queue Length", "lower is better"),
    ("throughput_veh", "Throughput", "higher is better"),
]
CONTROLLERS = ["fixed", "actuated", "ppo"]
LABELS = {"fixed": "Fixed", "actuated": "Actuated", "ppo": "PPO"}
COLOURS = {"fixed": "#4C72B0", "actuated": "#55A868", "ppo": "#C44E52"}


def load_metrics() -> pd.DataFrame:
    """Load and validate the Week 3 metrics CSV."""
    if not os.path.isfile(WEEK3_CSV):
        raise FileNotFoundError(f"Metrics CSV not found: {WEEK3_CSV}")
    df = pd.read_csv(WEEK3_CSV)
    missing = set(CONTROLLERS) - set(df["controller"].unique())
    if missing:
        raise ValueError(f"Missing Week 3 controller rows: {sorted(missing)}")
    return df


def build_chart(df: pd.DataFrame) -> None:
    """Render and save the 3-panel Week 3 comparison chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x_pos = np.arange(len(CONTROLLERS))
    fig.suptitle("Week 3 Corridor Controller Comparison · 3 seeds", fontsize=13, fontweight="bold")
    for ax, (column, title, direction) in zip(axes, METRICS):
        grouped = df.groupby("controller")[column]
        means = [grouped.mean()[controller] for controller in CONTROLLERS]
        stds = [grouped.std().fillna(0)[controller] for controller in CONTROLLERS]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=[COLOURS[c] for c in CONTROLLERS])
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + max(means) * 0.01, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title}\n({direction})", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS[c] for c in CONTROLLERS])
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    os.makedirs(os.path.dirname(WEEK3_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(WEEK3_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", WEEK3_PNG)


def write_report(df: pd.DataFrame) -> None:
    """Write a concise Week 3 report with seed variance."""
    grouped = df.groupby("controller")["avg_wait_time_s"].agg(["mean", "std"]).reindex(CONTROLLERS)
    lines = [
        "# Week 3 Report",
        "",
        "Scope: corridor-wide metrics while one selected corridor signal is controlled through the single-agent PPO environment.",
        "",
        "| Controller | Avg wait mean (s) | Avg wait std (s) |",
        "|---|---:|---:|",
    ]
    for controller, row in grouped.iterrows():
        lines.append(f"| {LABELS[controller]} | {row['mean']:.2f} | {row['std']:.2f} |")
    ppo = grouped.loc["ppo", "mean"]
    fixed = grouped.loc["fixed", "mean"]
    actuated = grouped.loc["actuated", "mean"]
    verdict = "PPO beat both fixed-time and actuated on avg_wait_time_s." if ppo < fixed and ppo < actuated else "PPO did not beat both baselines on avg_wait_time_s; further training or reward diagnosis is required."
    lines.extend(["", f"Verdict: {verdict}", "", "Honest observation: the actuated controller remains a strong baseline on this synthetic corridor, especially when demand is near saturation."])
    with open(WEEK3_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    log.info("Saved %s", WEEK3_REPORT)


def main() -> None:
    """CLI entry point."""
    df = load_metrics()
    build_chart(df)
    write_report(df)


if __name__ == "__main__":
    main()
