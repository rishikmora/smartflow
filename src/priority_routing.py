"""Week 9: emergency-vehicle signal preemption on the corridor.

Injects emergency vehicles into the committed corridor and, when preemption is
enabled, holds a green for whichever approach an emergency vehicle is closing
on. The point is to measure both halves of the trade: what preemption buys the
emergency vehicle, and what it costs everyone else. Reporting only the first
half would be the easy and dishonest version of this result.

Preemption is a rule, not a learned policy, so no training is involved. It sits
outside the RL loop and overrides the signal directly through TraCI, which is
also how a real preemption system works - it is a safety override, not an
optimiser.

Usage:
    python src/priority_routing.py
    python src/priority_routing.py --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traci

from metrics import MetricsCollector
from week4_config import CORRIDOR_NET, CORRIDOR_ROUTE, CORRIDOR_TLS_IDS
from week9_config import (
    EMERGENCY_COUNT,
    PREEMPT_RANGE_M,
    PRIORITY_CHART,
    PRIORITY_CSV,
    PRIORITY_RESULTS,
    PRIORITY_SEEDS,
    PRIORITY_SIM_SECONDS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SUMO_BIN = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
EMERGENCY_TYPE = "emergency"
CSV_HEADER = ["mode", "seed", "emergency_mean_wait_s", "emergency_mean_duration_s",
              "general_avg_wait_s", "general_throughput", "general_max_queue",
              "preemptions"]


def add_emergency_vehicles(count: int, sim_seconds: int, rng_seed: int) -> list[str]:
    """Insert emergency vehicles on existing routes, spread through the episode.

    Args:
        count: how many to insert.
        sim_seconds: episode length, used to space departures.
        rng_seed: seed for route choice.

    Returns:
        The ids of the vehicles inserted.
    """
    import random

    rng = random.Random(rng_seed)
    routes = list(traci.route.getIDList())
    if not routes:
        log.warning("No routes available; cannot insert emergency vehicles.")
        return []

    try:
        traci.vehicletype.copy("DEFAULT_VEHTYPE", EMERGENCY_TYPE)
        traci.vehicletype.setVehicleClass(EMERGENCY_TYPE, "emergency")
        traci.vehicletype.setColor(EMERGENCY_TYPE, (230, 40, 40, 255))
    except traci.TraCIException as exc:
        log.warning("Could not create the emergency vType: %s", exc)

    ids: list[str] = []
    # Start after warm-up so the corridor is loaded when they arrive.
    for index in range(count):
        depart = 300 + index * ((sim_seconds - 600) // max(count, 1))
        vehicle_id = f"emg_{index}"
        try:
            traci.vehicle.add(vehicle_id, rng.choice(routes),
                              typeID=EMERGENCY_TYPE, depart=str(depart))
            ids.append(vehicle_id)
        except traci.TraCIException as exc:
            log.warning("Could not add %s: %s", vehicle_id, exc)
    return ids


def preempt(tls_ids: list[str], active: dict[str, int]) -> int:
    """Hold a green for any approach an emergency vehicle is closing on.

    For each signal, the vehicles on its incoming lanes are checked; if one is an
    emergency vehicle within range, the signal is switched to a phase that gives
    its lane green and held there while it approaches.

    Args:
        tls_ids: signals to consider.
        active: per-signal record of the phase currently forced, mutated here.

    Returns:
        How many signals were preempted this step.
    """
    count = 0
    for tls_id in tls_ids:
        controlled = traci.trafficlight.getControlledLanes(tls_id)
        target_index: int | None = None

        for link_index, lane in enumerate(controlled):
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane):
                if not vehicle_id.startswith("emg_"):
                    continue
                try:
                    lane_length = traci.lane.getLength(lane)
                    position = traci.vehicle.getLanePosition(vehicle_id)
                except traci.TraCIException:
                    continue
                if lane_length - position <= PREEMPT_RANGE_M:
                    target_index = link_index
                    break
            if target_index is not None:
                break

        if target_index is None:
            if tls_id in active:
                # Hand the junction back to its own program.
                del active[tls_id]
            continue

        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        best_phase = None
        for phase_index, phase in enumerate(logic.phases):
            state = phase.state
            if target_index < len(state) and state[target_index] in "Gg":
                best_phase = phase_index
                break
        if best_phase is None:
            continue

        if active.get(tls_id) != best_phase:
            traci.trafficlight.setPhase(tls_id, best_phase)
            active[tls_id] = best_phase
        # Keep it there while the emergency vehicle is inside the window.
        traci.trafficlight.setPhaseDuration(tls_id, 5.0)
        count += 1
    return count


def run_case(preemption: bool, seed: int) -> dict[str, Any]:
    """Run one episode with or without preemption.

    Args:
        preemption: whether to enable signal preemption.
        seed: SUMO seed.

    Returns:
        A metrics row.
    """
    traci.start([
        SUMO_BIN, "-n", CORRIDOR_NET, "-r", CORRIDOR_ROUTE,
        "--no-step-log", "--no-warnings",
        "--waiting-time-memory", "1000", "--max-depart-delay", "-1",
        "--time-to-teleport", "-1", "--seed", str(seed),
    ])
    emergency_wait: dict[str, float] = {}
    emergency_depart: dict[str, float] = {}
    emergency_duration: dict[str, float] = {}
    preemptions = 0
    active: dict[str, int] = {}

    try:
        add_emergency_vehicles(EMERGENCY_COUNT, PRIORITY_SIM_SECONDS, seed)
        collector = MetricsCollector(traci)

        for step in range(PRIORITY_SIM_SECONDS):
            if preemption:
                preemptions += preempt(CORRIDOR_TLS_IDS, active)

            traci.simulationStep()
            collector.sample()

            now = traci.simulation.getTime()
            for vehicle_id in traci.vehicle.getIDList():
                if not vehicle_id.startswith("emg_"):
                    continue
                emergency_depart.setdefault(vehicle_id, now)
                if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                    emergency_wait[vehicle_id] = emergency_wait.get(vehicle_id, 0.0) + 1.0
            for vehicle_id in traci.simulation.getArrivedIDList():
                if vehicle_id.startswith("emg_") and vehicle_id in emergency_depart:
                    emergency_duration[vehicle_id] = now - emergency_depart[vehicle_id]

        corridor = collector.corridor_row()
    finally:
        traci.close()

    waits = list(emergency_wait.values()) or [0.0]
    durations = list(emergency_duration.values()) or [float("nan")]
    row = {
        "mode": "preemption" if preemption else "none",
        "seed": seed,
        "emergency_mean_wait_s": sum(waits) / len(waits),
        "emergency_mean_duration_s": sum(durations) / len(durations),
        "general_avg_wait_s": float(corridor["avg_wait_time_s"]),
        "general_throughput": float(corridor["throughput_veh"]),
        "general_max_queue": float(corridor["max_queue_len"]),
        "preemptions": preemptions,
    }
    log.info("  %-10s seed %d -> emergency wait %6.2fs | general wait %6.2fs "
             "thru %4d | %d preemption steps",
             row["mode"], seed, row["emergency_mean_wait_s"],
             row["general_avg_wait_s"], int(row["general_throughput"]), preemptions)
    return row


def plot(rows: list[dict[str, Any]]) -> None:
    """Chart the emergency benefit against the cost to general traffic.

    Args:
        rows: metric rows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = ["none", "preemption"]
    metrics = [("emergency_mean_wait_s", "Emergency vehicle wait (s)"),
               ("general_avg_wait_s", "General traffic wait (s)"),
               ("general_throughput", "General throughput (vehicles)")]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, (key, label) in zip(axes, metrics):
        means, spreads = [], []
        for mode in modes:
            values = [float(r[key]) for r in rows if r["mode"] == mode]
            means.append(float(np.mean(values)) if values else 0.0)
            spreads.append(float(np.std(values)) if values else 0.0)
        ax.bar(modes, means, yerr=spreads, capsize=4,
               color=["#8a929e", "#c7362f"], alpha=0.92)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Week 9 - emergency preemption: what it buys, and what it costs",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PRIORITY_CHART, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", PRIORITY_CHART)


def main() -> None:
    """Run the preemption comparison and report."""
    parser = argparse.ArgumentParser(description="Emergency-vehicle preemption.")
    parser.add_argument("--seeds", type=int, nargs="+", default=PRIORITY_SEEDS)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for preemption in (False, True):
        for seed in args.seeds:
            rows.append(run_case(preemption, seed))

    with open(PRIORITY_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", PRIORITY_CSV)
    plot(rows)

    def mean(mode: str, key: str) -> float:
        values = [float(r[key]) for r in rows if r["mode"] == mode]
        return sum(values) / len(values) if values else float("nan")

    emg_none, emg_pre = mean("none", "emergency_mean_wait_s"), mean("preemption", "emergency_mean_wait_s")
    gen_none, gen_pre = mean("none", "general_avg_wait_s"), mean("preemption", "general_avg_wait_s")
    emg_gain = (emg_none - emg_pre) / emg_none * 100 if emg_none else 0.0
    gen_cost = (gen_pre - gen_none) / gen_none * 100 if gen_none else 0.0

    log.info("")
    log.info("%d-seed means:", len(args.seeds))
    log.info("  emergency wait   %6.2f -> %6.2f s  (%+.1f%%)", emg_none, emg_pre, -emg_gain)
    log.info("  general wait     %6.2f -> %6.2f s  (%+.1f%%)", gen_none, gen_pre, gen_cost)
    log.info("  net: preemption %s emergency delay and %s general delay",
             "cuts" if emg_gain > 0 else "raises",
             "raises" if gen_cost > 0 else "cuts")

    payload = {
        "seeds": args.seeds,
        "emergency_count": EMERGENCY_COUNT,
        "preempt_range_m": PREEMPT_RANGE_M,
        "rows": rows,
        "summary": {
            "emergency_wait_none_s": emg_none,
            "emergency_wait_preempt_s": emg_pre,
            "emergency_improvement_pct": emg_gain,
            "general_wait_none_s": gen_none,
            "general_wait_preempt_s": gen_pre,
            "general_cost_pct": gen_cost,
        },
    }
    with open(PRIORITY_RESULTS, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("Wrote %s", PRIORITY_RESULTS)


if __name__ == "__main__":
    main()
