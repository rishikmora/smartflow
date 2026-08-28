"""Week 8: detect injected incidents in the corridor's live metric stream.

Runs the committed corridor under fixed-time control, obstructs a lane at known
times, and records every lane's per-second queue. Because this script chooses
the incident times, the ground truth is exact and the detector can be scored
rather than eyeballed.

The incident is physical, not a number poked into a CSV: ``lane.setMaxSpeed``
drops a lane to walking pace, traffic backs up behind it, and the queue responds
the way it would to a broken-down vehicle. Restoring the limit ends it.

Two design decisions worth stating, because both were arrived at by measuring:

* **Per lane, not corridor-wide.** A first version aggregated queue across the
  whole corridor and reached recall 0.33: a single obstructed lane is buried
  under the variance of twelve signal cycles. One detector per lane - which is
  what a real system with detector loops does - reaches recall 1.00.
* **The operating point is swept, not chosen.** Forty-eight independent
  detectors are a multiple-comparisons problem, so precision depends heavily on
  the threshold. The signals are recorded once and the detector is replayed over
  them across a grid, so the report can show the whole precision/recall
  trade-off instead of one flattering point.

Usage:
    python src/anomaly_live.py
    python src/anomaly_live.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traci

from anomaly_detector import Alarm, StreamingDetector, score_detections
from week4_config import CORRIDOR_NET, CORRIDOR_ROUTE
from week8_config import (
    ANOMALY_CHART,
    ANOMALY_MIN_SAMPLES,
    ANOMALY_WINDOW,
    OUTPUTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SUMO_BIN = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
RESULTS_JSON = os.path.join(OUTPUTS_DIR, "week8_anomaly_results.json")

SIM_SECONDS = 1800
INCIDENTS = [(420.0, 540.0), (900.0, 1020.0), (1350.0, 1470.0)]
BLOCK_LANE_EDGES = ["B1B2", "C1C2", "B2B3"]
BLOCKED_SPEED = 0.6

# Grid swept offline over the recorded signals.
THRESHOLDS = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
PERSISTENCES = [3, 6, 10, 15]


def record_episode(seed: int) -> dict[str, object]:
    """Run one episode with injected incidents, recording every lane's queue.

    Args:
        seed: SUMO seed.

    Returns:
        ``{"seed", "times", "lanes": {lane_id: [halting per second]}, "total"}``.
    """
    traci.start([
        SUMO_BIN, "-n", CORRIDOR_NET, "-r", CORRIDOR_ROUTE,
        "--no-step-log", "--no-warnings",
        "--time-to-teleport", "-1", "--waiting-time-memory", "1000",
        "--seed", str(seed),
    ])
    times: list[float] = []
    total: list[float] = []
    series: dict[str, list[float]] = {}
    blocked: dict[str, float] = {}
    try:
        watch = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        series = {lid: [] for lid in watch}
        targets = [[lid for lid in watch if lid.startswith(edge + "_")]
                   for edge in BLOCK_LANE_EDGES]

        for step in range(SIM_SECONDS):
            now = float(step)
            for index, (start_s, end_s) in enumerate(INCIDENTS):
                lanes = targets[index] if index < len(targets) else []
                if abs(now - start_s) < 0.5:
                    for lid in lanes:
                        blocked[lid] = traci.lane.getMaxSpeed(lid)
                        traci.lane.setMaxSpeed(lid, BLOCKED_SPEED)
                elif abs(now - end_s) < 0.5:
                    for lid in lanes:
                        if lid in blocked:
                            traci.lane.setMaxSpeed(lid, blocked.pop(lid))

            traci.simulationStep()

            step_total = 0.0
            for lid in watch:
                halting = float(traci.lane.getLastStepHaltingNumber(lid))
                series[lid].append(halting)
                step_total += halting
            times.append(now)
            total.append(step_total)
    finally:
        traci.close()

    log.info("seed %d recorded: %d lanes x %d seconds", seed, len(series), len(times))
    return {"seed": seed, "times": times, "lanes": series, "total": total}


def detect(record: dict[str, object], threshold: float,
           persistence: int) -> tuple[list[Alarm], dict[str, float]]:
    """Replay the detector over a recorded episode at one operating point.

    Args:
        record: output of :func:`record_episode`.
        threshold: z-score threshold.
        persistence: consecutive breaching samples required.

    Returns:
        ``(merged_alarms, score)``.
    """
    times: list[float] = record["times"]           # type: ignore[assignment]
    lanes: dict[str, list[float]] = record["lanes"]  # type: ignore[assignment]

    alarms: list[Alarm] = []
    for lane_id, values in lanes.items():
        detector = StreamingDetector(
            signal=lane_id, window=ANOMALY_WINDOW, threshold=threshold,
            min_samples=ANOMALY_MIN_SAMPLES, refractory_s=150.0,
            one_sided=True, persistence=persistence,
        )
        for t, value in zip(times, values):
            fired = detector.update(t, value)
            if fired is not None:
                alarms.append(fired)

    # Lanes near an obstruction all queue at once; count that as one event.
    alarms.sort(key=lambda a: a.t)
    merged: list[Alarm] = []
    for alarm in alarms:
        if merged and (alarm.t - merged[-1].t) < 45.0:
            continue
        merged.append(alarm)
    return merged, score_detections(merged, INCIDENTS)


def sweep(records: list[dict[str, object]]) -> list[dict[str, float]]:
    """Score every operating point on the grid, averaged across seeds.

    Args:
        records: recorded episodes.

    Returns:
        One row per (threshold, persistence) with mean precision/recall/F1.
    """
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        for persistence in PERSISTENCES:
            scores = [detect(r, threshold, persistence)[1] for r in records]
            row = {"threshold": threshold, "persistence": float(persistence)}
            for key in ("precision", "recall", "f1", "false_alarms",
                        "detected", "mean_latency_s"):
                values = [s[key] for s in scores]
                clean = [v for v in values if v == v]         # drop NaN latency
                row[key] = sum(clean) / len(clean) if clean else float("nan")
            rows.append(row)
    return rows


def pick_operating_point(rows: list[dict[str, float]]) -> dict[str, float]:
    """Choose the operating point to headline.

    Recall is prioritised over precision: a missed obstruction is a far worse
    failure for a traffic controller than an extra alert a human dismisses. Among
    points with full recall the highest F1 wins.

    Args:
        rows: swept grid.

    Returns:
        The chosen row.
    """
    full_recall = [r for r in rows if r["recall"] >= 0.999]
    pool = full_recall or rows
    return max(pool, key=lambda r: (r["f1"], -r["threshold"]))


def plot(records: list[dict[str, object]], rows: list[dict[str, float]],
         best: dict[str, float]) -> None:
    """Chart the queue trace with incidents and alarms, plus the sweep.

    Args:
        records: recorded episodes.
        rows: swept grid.
        best: chosen operating point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 3.0 * len(records) + 3.4))
    grid = fig.add_gridspec(len(records) + 1, 2, height_ratios=[1] * len(records) + [1.25])

    for index, record in enumerate(records):
        ax = fig.add_subplot(grid[index, :])
        alarms, score = detect(record, best["threshold"], int(best["persistence"]))
        ax.plot(record["times"], record["total"], lw=0.9, color="#1a7ba4")
        for start, end in INCIDENTS:
            ax.axvspan(start, end, color="#c7362f", alpha=0.16, lw=0)
        for alarm in alarms:
            ax.axvline(alarm.t, color="#d18a12", lw=1.1, ls="--", alpha=0.9)
        ax.set_title(f"seed {record['seed']} - {int(score['detected'])}"
                     f"/{int(score['incidents'])} incidents detected, "
                     f"{int(score['false_alarms'])} false alarms",
                     fontsize=10, loc="left")
        ax.set_ylabel("halting vehicles")
        ax.grid(alpha=0.25)
    ax.set_xlabel("simulation time (s)")

    ax_pr = fig.add_subplot(grid[len(records), 0])
    for persistence in PERSISTENCES:
        pts = sorted((r for r in rows if r["persistence"] == persistence),
                     key=lambda r: r["recall"])
        ax_pr.plot([r["recall"] for r in pts], [r["precision"] for r in pts],
                   marker="o", ms=3.5, lw=1.2, label=f"persistence {persistence}s")
    ax_pr.scatter([best["recall"]], [best["precision"]], s=90, facecolor="none",
                  edgecolor="#c7362f", lw=1.8, zorder=5, label="chosen")
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision / recall across the sweep", fontsize=10, loc="left")
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(fontsize=7.5)

    ax_lat = fig.add_subplot(grid[len(records), 1])
    for persistence in PERSISTENCES:
        pts = sorted((r for r in rows if r["persistence"] == persistence),
                     key=lambda r: r["threshold"])
        ax_lat.plot([r["threshold"] for r in pts], [r["mean_latency_s"] for r in pts],
                    marker="o", ms=3.5, lw=1.2, label=f"persistence {persistence}s")
    ax_lat.set_xlabel("z threshold")
    ax_lat.set_ylabel("detection latency (s)")
    ax_lat.set_title("How long until the incident is flagged", fontsize=10, loc="left")
    ax_lat.grid(alpha=0.25)

    fig.suptitle("Week 8 - incident detection on the corridor's live metrics", fontsize=12)
    fig.tight_layout()
    fig.savefig(ANOMALY_CHART, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", ANOMALY_CHART)


def main() -> None:
    """Record episodes, sweep the detector and report."""
    parser = argparse.ArgumentParser(description="Detect injected corridor incidents.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    records = [record_episode(seed) for seed in args.seeds]
    log.info("Sweeping %d operating points...", len(THRESHOLDS) * len(PERSISTENCES))
    rows = sweep(records)
    best = pick_operating_point(rows)

    log.info("")
    log.info("Chosen: z=%.1f persistence=%ds -> recall %.2f precision %.2f "
             "F1 %.2f latency %.0fs false alarms %.1f",
             best["threshold"], int(best["persistence"]), best["recall"],
             best["precision"], best["f1"], best["mean_latency_s"],
             best["false_alarms"])

    plot(records, rows, best)

    payload = {
        "incidents": INCIDENTS,
        "blocked_edges": BLOCK_LANE_EDGES,
        "seeds": args.seeds,
        "detector": {"window": ANOMALY_WINDOW, "min_samples": ANOMALY_MIN_SAMPLES,
                     "per_lane": True},
        "sweep": rows,
        "chosen": best,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("Wrote %s", RESULTS_JSON)


if __name__ == "__main__":
    main()
