"""
Benchmark evaluation for the 2-way single-intersection scenario.

Handles both Day 3 (fixed-time baseline, 3 seeds) and Day 5 (PPO policy, 3 seeds).

Usage:
    # Day 3 — fixed-time baseline
    python src/eval_benchmark.py --controller fixed --seeds 0 1 2

    # Day 5 — PPO evaluation (models must exist from Day 4)
    python src/eval_benchmark.py --controller ppo --seeds 0 1 2

Appends to outputs/week2_benchmark_metrics.csv.
CSV schema: controller, seed, avg_wait_time_s, max_queue_len, throughput_veh
"""
import argparse
import csv
import logging
import os
import sys

import traci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

from week2_config import (
    BENCHMARK_CSV,
    BENCHMARK_NET,
    BENCHMARK_ROUTE,
    CSV_HEADER,
    DELTA_TIME,
    MAX_GREEN,
    MIN_GREEN,
    OUTPUTS_DIR,
    SIM_SECONDS,
    TRAIN_SEEDS,
    YELLOW_TIME,
    model_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_env(fixed_ts: bool, seed: int) -> SumoEnvironment:
    """Create the benchmark environment.

    Args:
        fixed_ts: if True the signal timing is not controlled by the agent.
        seed: SUMO random seed.

    Returns:
        Configured SumoEnvironment.
    """
    return SumoEnvironment(
        net_file=BENCHMARK_NET,
        route_file=BENCHMARK_ROUTE,
        num_seconds=SIM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        single_agent=True,
        use_gui=False,
        sumo_warnings=False,
        sumo_seed=seed,
        fixed_ts=fixed_ts,
        add_system_info=True,
    )


def _collect_metrics(env: SumoEnvironment, seed: int, controller: str,
                     model: "PPO | None" = None) -> dict:
    """Run one full episode and return metrics aligned with Week 1's schema.

    Metrics are collected by querying traci directly after each env.step().

    Definitions:
        avg_wait_time_s  — vehicle-seconds stopped / arrived vehicles
        max_queue_len    — peak simultaneous halted vehicles
        throughput_veh   — vehicles that completed their route

    NOTE: env.step() advances SUMO by delta_time=5 sub-steps internally.
    traci.simulation.getArrivedIDList() only returns arrivals from the LAST
    sub-step, so we detect departures via absence from getIDList() instead.
    This captures all arrivals regardless of which sub-step they occurred in.

    Args:
        env: a freshly reset SumoEnvironment (single_agent=True).
        seed: passed to env.reset(); also used in the returned row.
        controller: label for the CSV row ("fixed" or "ppo").
        model: loaded PPO model; if None, action_space.sample() is used for
               fixed_ts=True environments (action is ignored by SUMO).

    Returns:
        Dict with keys matching CSV_HEADER.
    """
    obs, _ = env.reset(seed=seed)

    seen_vehicles: set[str] = set()
    veh_stopped_s: dict[str, int] = {}
    completed_waits: list[int] = []
    max_queue = 0

    terminated = truncated = False
    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()   # ignored by fixed_ts env

        obs, _, terminated, truncated, _ = env.step(int(action))

        # --- traci-level metric collection ---
        veh_ids: set[str] = set(traci.vehicle.getIDList())
        seen_vehicles.update(veh_ids)

        for vid in veh_ids:
            if traci.vehicle.getWaitingTime(vid) > 0:
                veh_stopped_s[vid] = veh_stopped_s.get(vid, 0) + 1

        # Vehicles absent from current list but tracked → have departed/arrived
        departed = set(veh_stopped_s.keys()) - veh_ids
        for vid in departed:
            completed_waits.append(veh_stopped_s.pop(vid))

        halted = sum(
            traci.lane.getLastStepHaltingNumber(lid)
            for lid in traci.lane.getIDList()
        )
        if halted > max_queue:
            max_queue = halted

    still_in_network = len(traci.vehicle.getIDList())
    throughput = len(seen_vehicles) - still_in_network
    avg_wait = sum(completed_waits) / len(completed_waits) if completed_waits else 0.0

    return {
        "controller":      controller,
        "seed":            seed,
        "avg_wait_time_s": round(avg_wait, 2),
        "max_queue_len":   max_queue,
        "throughput_veh":  throughput,
    }


def _save_row(row: dict) -> None:
    """Append one metrics row to BENCHMARK_CSV (writes header on first call).

    Args:
        row: dict with keys matching CSV_HEADER.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    write_header = not os.path.isfile(BENCHMARK_CSV)
    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info(
        "Logged  controller=%-6s  seed=%d  wait=%.1fs  queue=%d  through=%d",
        row["controller"], row["seed"],
        row["avg_wait_time_s"], row["max_queue_len"], row["throughput_veh"],
    )


def run_fixed(seeds: list[int]) -> None:
    """Evaluate the fixed-time (static tlLogic) controller on given seeds.

    Args:
        seeds: list of SUMO random seeds to evaluate.
    """
    log.info("=== Fixed-time baseline (%d seeds) ===", len(seeds))
    for seed in seeds:
        log.info("Running seed=%d ...", seed)
        env = _build_env(fixed_ts=True, seed=seed)
        try:
            row = _collect_metrics(env, seed=seed, controller="fixed")
        finally:
            env.close()
        _save_row(row)


def run_ppo(seeds: list[int]) -> None:
    """Evaluate trained PPO models deterministically on given seeds.

    Args:
        seeds: list of seeds; each must have a corresponding model zip.

    Raises:
        FileNotFoundError: if a model checkpoint is missing.
    """
    log.info("=== PPO evaluation (%d seeds) ===", len(seeds))
    for seed in seeds:
        mp = model_path(seed)
        if not os.path.isfile(mp):
            raise FileNotFoundError(
                f"Model not found: {mp}\n"
                "Run train_ppo_benchmark.py --seed {seed} first."
            )
        log.info("Loading model: %s", mp)
        model = PPO.load(mp)

        log.info("Evaluating seed=%d ...", seed)
        env = _build_env(fixed_ts=False, seed=seed)
        try:
            row = _collect_metrics(env, seed=seed, controller="ppo", model=model)
        finally:
            env.close()
        _save_row(row)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-time or PPO controller on the benchmark."
    )
    parser.add_argument(
        "--controller", choices=["fixed", "ppo"], required=True,
        help="Which controller to evaluate.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=TRAIN_SEEDS,
        help="SUMO seeds to evaluate (default: 0 1 2).",
    )
    args = parser.parse_args()

    if args.controller == "fixed":
        run_fixed(args.seeds)
    else:
        run_ppo(args.seeds)


if __name__ == "__main__":
    main()
