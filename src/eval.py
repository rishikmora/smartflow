"""
Benchmark harness for SmartFlow.
Usage:
    python src/eval.py --controller fixed    --seed 0
    python src/eval.py --controller actuated --seed 0

Appends one row to outputs/metrics.csv (writes header only on first run).
CSV schema: controller, seed, avg_wait_time_s, max_queue_len, throughput_veh, total_co2_kg
"""
import argparse
import csv
import os
import sys

SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    print("ERROR: SUMO_HOME not set")
    sys.exit(1)

SUMO_BIN = os.path.join(SUMO_HOME, "bin", "sumo.exe")

import traci  # noqa: E402

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
SIM_END = 1800  # simulated seconds — matches route file duration

CONFIGS = {
    "fixed":    os.path.join(ROOT, "configs", "corridor.sumocfg"),
    "actuated": os.path.join(ROOT, "configs", "corridor_actuated.sumocfg"),
}

METRICS_CSV = os.path.join(ROOT, "outputs", "metrics.csv")
CSV_HEADER  = ["controller", "seed",
               "avg_wait_time_s", "max_queue_len",
               "throughput_veh", "total_co2_kg"]


def run(controller: str, seed: int) -> dict:
    cfg = CONFIGS[controller]
    if not os.path.isfile(cfg):
        print(f"ERROR: config not found: {cfg}")
        sys.exit(1)

    traci.start([
        SUMO_BIN, "-c", cfg,
        "--seed", str(seed),
        "--no-step-log", "--no-warnings",
    ])

    # per-vehicle accumulated stopped seconds, cleared when each vehicle arrives
    veh_stopped_s: dict[str, int] = {}
    completed_waits: list[int]    = []

    total_co2_mg = 0.0
    max_queue    = 0
    arrived      = 0

    lane_ids = traci.lane.getIDList()

    while traci.simulation.getTime() < SIM_END:
        traci.simulationStep()

        veh_ids = traci.vehicle.getIDList()

        for vid in veh_ids:
            if traci.vehicle.getWaitingTime(vid) > 0:
                veh_stopped_s[vid] = veh_stopped_s.get(vid, 0) + 1
            total_co2_mg += traci.vehicle.getCO2Emission(vid)  # mg/s × 1 s = mg

        # capture completed vehicles' waits before they're removed
        for vid in traci.simulation.getArrivedIDList():
            completed_waits.append(veh_stopped_s.pop(vid, 0))
            arrived += 1

        queue = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids)
        if queue > max_queue:
            max_queue = queue

    traci.close()

    avg_wait = (sum(completed_waits) / len(completed_waits)
                if completed_waits else 0.0)

    return {
        "controller":      controller,
        "seed":            seed,
        "avg_wait_time_s": round(avg_wait, 2),
        "max_queue_len":   max_queue,
        "throughput_veh":  arrived,
        "total_co2_kg":    round(total_co2_mg / 1e6, 4),
    }


def save(row: dict) -> None:
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    write_header = not os.path.isfile(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged -> {METRICS_CSV}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["fixed", "actuated"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    print(f"Running controller={args.controller} seed={args.seed} …")
    row = run(args.controller, args.seed)
    save(row)
