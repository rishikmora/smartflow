"""Week 8 scenario planner: lane closures and weather on the committed corridor.

Answers the planning question the roadmap asks for - "what happens to this
corridor if we close a link, or if it rains?" - by running the real network
under each combination and measuring it with the same per-second collector every
other week uses, so the numbers sit on the same axis as Weeks 1-6.

**Closures** disallow every vehicle class on an edge's lanes and make the edge
look expensive to the router, then reroute traffic that was heading for it.
Blocking without rerouting would only measure how fast a corridor gridlocks.

**Weather** is modelled the way SUMO itself recommends: not as graphics, but as
the driving-behaviour changes rain and fog actually cause - lower speeds, longer
headways, gentler braking and bigger gaps. Values are multipliers held in
``week8_config.WEATHER``.

Usage:
    python src/scenario_planner.py
    python src/scenario_planner.py --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traci

from metrics import MetricsCollector
from week4_config import CORRIDOR_NET, CORRIDOR_ROUTE
from week8_config import (
    CLOSURES,
    CLOSURE_SPEED,
    PLANNER_CHART,
    PLANNER_CSV,
    PLANNER_SEEDS,
    PLANNER_SIM_SECONDS,
    WEATHER,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SUMO_BIN = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
CSV_HEADER = ["closure", "weather", "seed", "avg_wait_time_s", "max_queue_len",
              "throughput_veh", "total_co2_kg", "completed_trips", "blocked_edge"]

# Vehicles are rerouted periodically rather than every step: rerouting all of
# them every second is expensive and unrealistic - drivers do not re-plan
# continuously.
REROUTE_EVERY_S = 60


def apply_weather(vehicle_id: str, weather: str) -> None:
    """Apply a weather profile's driving-behaviour changes to one vehicle.

    Args:
        vehicle_id: SUMO vehicle id.
        weather: key into :data:`WEATHER`.
    """
    profile = WEATHER[weather]
    if weather == "clear":
        return
    try:
        base_factor = traci.vehicle.getSpeedFactor(vehicle_id)
        traci.vehicle.setSpeedFactor(vehicle_id, base_factor * profile["speed_factor"])
        traci.vehicle.setTau(vehicle_id, traci.vehicle.getTau(vehicle_id) * profile["tau"])
        traci.vehicle.setDecel(vehicle_id,
                               max(0.5, traci.vehicle.getDecel(vehicle_id) * profile["decel"]))
        traci.vehicle.setMinGap(vehicle_id,
                                traci.vehicle.getMinGap(vehicle_id) * profile["min_gap"])
    except traci.TraCIException:
        # The vehicle can leave between listing and configuring it.
        pass


def restrict_edge(edge_id: str) -> list[str]:
    """Reduce an edge to walking pace and make it expensive to route over.

    Roadworks rather than a total ban: ``setDisallowed`` was tried first and
    invalidated the routes of every vehicle whose trip *ends* on this edge,
    which SUMO reports as "no valid route" and which measures cancelled trips
    instead of rerouting cost.

    Args:
        edge_id: the edge to restrict.

    Returns:
        The lane ids that were restricted.
    """
    lanes = [lid for lid in traci.lane.getIDList()
             if not lid.startswith(":") and lid.rsplit("_", 1)[0] == edge_id]
    for lane_id in lanes:
        traci.lane.setMaxSpeed(lane_id, CLOSURE_SPEED)
    # A very large travel time means any alternative route wins.
    traci.edge.adaptTraveltime(edge_id, 10_000.0)
    return lanes


def run_case(closure: str, blocked_edge: str | None, weather: str,
             seed: int) -> dict[str, object]:
    """Run one closure/weather/seed combination end to end.

    Args:
        closure: closure label.
        blocked_edge: edge to close, or None.
        weather: key into :data:`WEATHER`.
        seed: SUMO seed.

    Returns:
        A metrics row.
    """
    traci.start([
        SUMO_BIN, "-n", CORRIDOR_NET, "-r", CORRIDOR_ROUTE,
        "--no-step-log", "--no-warnings",
        "--waiting-time-memory", "1000", "--max-depart-delay", "-1",
        "--time-to-teleport", "300",
        "--seed", str(seed),
        # Rerouting devices let traffic respond to the restriction the way a
        # navigation system would, instead of driving into it regardless.
        "--device.rerouting.probability", "1",
        "--device.rerouting.period", "60",
        "--device.rerouting.adaptation-interval", "10",
    ])
    try:
        closed_lanes: list[str] = []
        if blocked_edge:
            closed_lanes = restrict_edge(blocked_edge)

        collector = MetricsCollector(traci)
        dressed: set[str] = set()

        for step in range(PLANNER_SIM_SECONDS):
            for vid in traci.vehicle.getIDList():
                if vid not in dressed:
                    dressed.add(vid)
                    apply_weather(vid, weather)

            traci.simulationStep()
            collector.sample()

        row = collector.corridor_row()
    finally:
        traci.close()

    record = {
        "closure": closure,
        "weather": weather,
        "seed": seed,
        "avg_wait_time_s": row["avg_wait_time_s"],
        "max_queue_len": row["max_queue_len"],
        "throughput_veh": row["throughput_veh"],
        "total_co2_kg": row["total_co2_kg"],
        "completed_trips": row["throughput_veh"],
        "blocked_edge": blocked_edge or "",
    }
    log.info("  %-13s %-5s seed %d -> wait %6.2fs  queue %4d  thru %4d  "
             "closed %d lanes",
             closure, weather, seed, record["avg_wait_time_s"],
             int(record["max_queue_len"]), int(record["throughput_veh"]),
             len(closed_lanes))
    return record


def plot(rows: list[dict[str, object]]) -> None:
    """Chart wait, throughput and CO2 across the scenario grid.

    Args:
        rows: metric rows from :func:`run_case`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    closures = [c[0] for c in CLOSURES]
    weathers = list(WEATHER)
    metrics = [("avg_wait_time_s", "Average wait (s)"),
               ("throughput_veh", "Throughput (vehicles)"),
               ("total_co2_kg", "CO2 (kg)")]
    colours = {"clear": "#1a7ba4", "rain": "#d18a12", "fog": "#8a5fa8"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    width = 0.26
    for ax, (key, label) in zip(axes, metrics):
        for index, weather in enumerate(weathers):
            values, errors = [], []
            for closure in closures:
                subset = [float(r[key]) for r in rows
                          if r["closure"] == closure and r["weather"] == weather]
                mean = sum(subset) / len(subset) if subset else 0.0
                spread = (max(subset) - min(subset)) / 2 if len(subset) > 1 else 0.0
                values.append(mean)
                errors.append(spread)
            positions = [i + (index - 1) * width for i in range(len(closures))]
            ax.bar(positions, values, width, yerr=errors, capsize=3,
                   color=colours[weather], label=weather, alpha=0.92)
        ax.set_xticks(range(len(closures)))
        ax.set_xticklabels(closures)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(title="weather", fontsize=8)
    fig.suptitle("Week 8 scenario planner - closures x weather, 3-seed means "
                 "(bars: min-max/2)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLANNER_CHART, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", PLANNER_CHART)


def main() -> None:
    """Run the full scenario grid and write results."""
    parser = argparse.ArgumentParser(description="Run the Week 8 scenario planner.")
    parser.add_argument("--seeds", type=int, nargs="+", default=PLANNER_SEEDS)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    total = len(CLOSURES) * len(WEATHER) * len(args.seeds)
    log.info("Running %d cases (%d closures x %d weather x %d seeds)",
             total, len(CLOSURES), len(WEATHER), len(args.seeds))

    for closure, edge, _description in CLOSURES:
        for weather in WEATHER:
            for seed in args.seeds:
                rows.append(run_case(closure, edge, weather, seed))

    os.makedirs(os.path.dirname(PLANNER_CSV), exist_ok=True)
    with open(PLANNER_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s (%d rows)", PLANNER_CSV, len(rows))

    plot(rows)

    log.info("")
    log.info("%-13s %-6s %10s %10s %10s", "closure", "weather", "wait", "thru", "co2")
    for closure, _edge, _d in CLOSURES:
        for weather in WEATHER:
            subset = [r for r in rows if r["closure"] == closure and r["weather"] == weather]
            if not subset:
                continue
            mean = lambda k: sum(float(r[k]) for r in subset) / len(subset)  # noqa: E731
            log.info("%-13s %-6s %10.2f %10.0f %10.1f", closure, weather,
                     mean("avg_wait_time_s"), mean("throughput_veh"),
                     mean("total_co2_kg"))


if __name__ == "__main__":
    main()
