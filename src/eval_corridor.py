"""Evaluate fixed, actuated, or PPO controllers for Week 3 corridor metrics."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_config import CORRIDOR_ACTUATED_CONFIG, CORRIDOR_FIXED_CONFIG, CSV_HEADER
from week3_config import MODELS_DIR, TRAIN_SEEDS, WEEK3_CSV, CORRIDOR_TLS_ID, model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def _sumo_binary() -> str:
    """Return the configured SUMO binary path."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set.")
    return os.path.join(sumo_home, "bin", "sumo.exe")


def _save_row(row: dict[str, Any]) -> None:
    """Append a metrics row to the Week 3 CSV."""
    os.makedirs(os.path.dirname(WEEK3_CSV), exist_ok=True)
    write_header = not os.path.isfile(WEEK3_CSV)
    with open(WEEK3_CSV, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _collect_traci_metrics(controller: str, seed: int, config_path: str) -> dict[str, Any]:
    """Run a full SUMO episode and collect corridor-wide metrics."""
    import traci

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"SUMO config not found: {config_path}")
    traci.start([_sumo_binary(), "-c", config_path, "--seed", str(seed), "--no-step-log", "--no-warnings"])
    veh_stopped_s: dict[str, int] = {}
    completed_waits: list[int] = []
    max_queue = 0
    arrived = 0
    total_co2_mg = 0.0
    lane_ids = traci.lane.getIDList()
    try:
        while traci.simulation.getTime() < 1800:
            traci.simulationStep()
            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getWaitingTime(vid) > 0:
                    veh_stopped_s[vid] = veh_stopped_s.get(vid, 0) + 1
                total_co2_mg += traci.vehicle.getCO2Emission(vid)
            for vid in traci.simulation.getArrivedIDList():
                completed_waits.append(veh_stopped_s.pop(vid, 0))
                arrived += 1
            max_queue = max(max_queue, sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lane_ids))
    finally:
        traci.close()
    avg_wait = sum(completed_waits) / len(completed_waits) if completed_waits else 0.0
    return {
        "controller": controller,
        "seed": seed,
        "scope": "corridor-wide",
        "tls_id": CORRIDOR_TLS_ID,
        "avg_wait_time_s": round(avg_wait, 2),
        "max_queue_len": max_queue,
        "throughput_veh": arrived,
        "total_co2_kg": round(total_co2_mg / 1e6, 4),
    }


def _ppo_model_path(seed: int, tag: str = "") -> str:
    """Return the PPO model path for a seed and optional training tag."""
    if not tag:
        return model_path(seed)
    return os.path.join(MODELS_DIR, f"ppo_corridor_seed{seed}_{tag}.zip")


def _collect_ppo_metrics(seed: int, tag: str = "") -> dict[str, Any]:
    """Evaluate a trained PPO model deterministically through sumo-rl."""
    if not os.environ.get("SUMO_HOME"):
        raise EnvironmentError("SUMO_HOME is not set.")
    import traci
    from stable_baselines3 import PPO
    from sumo_rl import SumoEnvironment
    from week3_config import CORRIDOR_NET, CORRIDOR_ROUTE, DELTA_TIME, MAX_GREEN, MIN_GREEN, SIM_SECONDS, YELLOW_TIME

    path = _ppo_model_path(seed, tag)
    if not os.path.isfile(path):
        tag_hint = f" with tag '{tag}'" if tag else ""
        raise FileNotFoundError(f"Missing PPO model for seed {seed}{tag_hint}: {path}")
    env = SumoEnvironment(
        net_file=CORRIDOR_NET,
        route_file=CORRIDOR_ROUTE,
        out_csv_name=None,
        num_seconds=SIM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        single_agent=True,
        use_gui=False,
        sumo_warnings=False,
        sumo_seed=seed,
        reward_fn="diff-waiting-time",
        add_system_info=True,
    )
    model = PPO.load(path)
    veh_stopped_s: dict[str, int] = {}
    completed_waits: list[int] = []
    seen_vehicles: set[str] = set()
    max_queue = 0
    total_co2_mg = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            veh_ids = set(traci.vehicle.getIDList())
            seen_vehicles.update(veh_ids)
            for vid in veh_ids:
                if traci.vehicle.getWaitingTime(vid) > 0:
                    veh_stopped_s[vid] = veh_stopped_s.get(vid, 0) + 1
                total_co2_mg += traci.vehicle.getCO2Emission(vid)
            departed = set(veh_stopped_s.keys()) - veh_ids
            for vid in departed:
                completed_waits.append(veh_stopped_s.pop(vid))
            max_queue = max(max_queue, sum(traci.lane.getLastStepHaltingNumber(lid) for lid in traci.lane.getIDList()))
    finally:
        env.close()
    throughput = len(seen_vehicles) - len(traci.vehicle.getIDList()) if traci.isLoaded() else len(completed_waits)
    avg_wait = sum(completed_waits) / len(completed_waits) if completed_waits else 0.0
    return {
        "controller": "ppo",
        "seed": seed,
        "scope": "corridor-wide",
        "tls_id": CORRIDOR_TLS_ID,
        "avg_wait_time_s": round(avg_wait, 2),
        "max_queue_len": max_queue,
        "throughput_veh": throughput,
        "total_co2_kg": round(total_co2_mg / 1e6, 4),
    }


def run(controller: str, seeds: list[int], tag: str = "") -> None:
    """Run the requested controller for each seed."""
    for seed in seeds:
        if controller == "fixed":
            row = _collect_traci_metrics("fixed", seed, CORRIDOR_FIXED_CONFIG)
        elif controller == "actuated":
            row = _collect_traci_metrics("actuated", seed, CORRIDOR_ACTUATED_CONFIG)
        else:
            row = _collect_ppo_metrics(seed, tag)
            if tag:
                row["controller"] = f"ppo_{tag}"
        _save_row(row)
        log.info("Logged %s seed=%d wait=%.2f queue=%s throughput=%s", controller, seed, row["avg_wait_time_s"], row["max_queue_len"], row["throughput_veh"])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Week 3 corridor controllers.")
    parser.add_argument("--controller", choices=["fixed", "actuated", "ppo"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional PPO model tag, e.g. 'short' for ppo_corridor_seed0_short.zip.",
    )
    args = parser.parse_args()
    run(args.controller, args.seeds, args.tag)


if __name__ == "__main__":
    main()
