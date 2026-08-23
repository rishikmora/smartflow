"""Week 3 evaluation harness: fixed / actuated / actuated_single / PPO on the corridor.

Every controller is measured by the same :class:`metrics.MetricsCollector`, sampled
once per *simulated second*, at two scopes:

``junction``
    The controlled junction's incoming lanes. This is Week 3's primary claim — it is
    the only place a single RL agent can plausibly change anything.
``corridor``
    The whole 12-junction network, so the Week 1 numbers stay comparable.

Baselines drive raw TraCI. The PPO controller runs inside sumo-rl, whose ``step()``
advances several simulated seconds at once; the collector is therefore attached via
``ControlledSumoEnvironment``'s per-second ``step_hook`` rather than being called once
per agent decision. Sampling per decision would divide every waiting time by
``delta_time`` and silently flatter the RL controller.

Usage:
    python src/eval_corridor.py --controller fixed --seeds 0 1 2
    python src/eval_corridor.py --controller actuated --seeds 0 1 2
    python src/eval_corridor.py --controller actuated_single --seeds 0 1 2
    python src/eval_corridor.py --controller ppo --seeds 0 1 2

Appends rows to ``outputs/week3_corridor_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricsCollector
from smartflow_env import baseline_sumo_cmd, make_single_agent_env, require_sumo_home
from week3_config import (
    BASELINE_NETS,
    CORRIDOR_ACTUATED_CONFIG,
    CORRIDOR_FIXED_CONFIG,
    CORRIDOR_NET,
    CORRIDOR_ROUTE,
    CORRIDOR_SINGLE_ACTUATED_CONFIG,
    CORRIDOR_TLS_ID,
    CSV_HEADER,
    DELTA_TIME,
    MAX_GREEN,
    MIN_GREEN,
    SIM_SECONDS,
    TRAIN_SEEDS,
    WEEK3_CSV,
    YELLOW_TIME,
    model_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

BASELINE_CONFIGS = {
    "fixed": CORRIDOR_FIXED_CONFIG,
    "actuated": CORRIDOR_ACTUATED_CONFIG,
    "actuated_single": CORRIDOR_SINGLE_ACTUATED_CONFIG,
}

CONTROLLERS = ["fixed", "actuated", "actuated_single", "ppo"]


def _save_rows(rows: list[dict[str, Any]], csv_path: str = WEEK3_CSV) -> None:
    """Append metric rows to the Week 3 CSV, writing the header on first use.

    Args:
        rows: dicts whose keys are a subset of ``CSV_HEADER``.
        csv_path: destination CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADER, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise OSError(f"Could not append metrics to {csv_path}: {exc}. Is the file open elsewhere?") from exc


def _rows_from(collector: MetricsCollector, controller: str, seed: int) -> list[dict[str, Any]]:
    """Turn a finished collector into one junction-scope and one corridor-scope row."""
    junction = collector.junction_row()
    corridor = collector.corridor_row()
    return [
        {
            "controller": controller,
            "seed": seed,
            "scope": "junction",
            "tls_id": CORRIDOR_TLS_ID,
            "total_co2_kg": "",  # not attributable to a single junction
            **junction,
        },
        {
            "controller": controller,
            "seed": seed,
            "scope": "corridor",
            "tls_id": "all",
            **corridor,
        },
    ]


def eval_baseline(controller: str, seed: int) -> list[dict[str, Any]]:
    """Run a non-RL controller for one episode and return its metric rows.

    Args:
        controller: ``"fixed"``, ``"actuated"`` or ``"actuated_single"``.
        seed: SUMO RNG seed.

    Returns:
        Two rows (junction scope, corridor scope).

    Raises:
        FileNotFoundError: if the controller's ``.sumocfg`` is missing.
    """
    require_sumo_home()
    import traci

    cfg = BASELINE_CONFIGS[controller]
    if not os.path.isfile(cfg):
        raise FileNotFoundError(
            f"SUMO config not found for controller '{controller}': {cfg}. "
            "Run src/make_single_actuated_net.py if this is the actuated_single config."
        )

    traci.start(baseline_sumo_cmd(config=cfg, seed=seed))
    try:
        junction_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(CORRIDOR_TLS_ID)))
        collector = MetricsCollector(traci, junction_lanes=junction_lanes)
        while traci.simulation.getTime() < SIM_SECONDS:
            traci.simulationStep()
            collector.sample()
    finally:
        traci.close()
    return _rows_from(collector, controller, seed)


def eval_ppo(seed: int, tag: str = "", model_file: str | None = None,
             controller_label: str | None = None) -> list[dict[str, Any]]:
    """Run a trained PPO policy deterministically for one episode.

    Only ``CORRIDOR_TLS_ID`` is under RL control; the other eleven junctions run the
    fixed-time program from the network file, matching the ``fixed`` baseline.

    Args:
        seed: SUMO RNG seed (and, unless ``model_file`` is given, the model selector).
        tag: optional model tag, e.g. ``"short"``.
        model_file: evaluate this exact ``.zip`` instead of the seed's saved model.
            Used by ``select_best_checkpoint.py`` to score intermediate checkpoints.
        controller_label: override the ``controller`` value written to the CSV.

    Returns:
        Two rows (junction scope, corridor scope).

    Raises:
        FileNotFoundError: if the requested model does not exist.
    """
    require_sumo_home()
    try:
        from stable_baselines3 import PPO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not import stable_baselines3: {exc}") from exc

    path = model_file or model_path(seed, tag)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No trained PPO model at {path}. Train it first with "
            f"'python src/train_ppo_corridor.py --seed {seed}"
            + (f" --tag {tag}'" if tag else "'")
        )

    env = make_single_agent_env(
        net_file=CORRIDOR_NET,
        route_file=CORRIDOR_ROUTE,
        controlled_ts=CORRIDOR_TLS_ID,
        num_seconds=SIM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        seed=seed,
        reward_fn="diff-waiting-time",
        add_system_info=False,
    )
    model = PPO.load(path)
    try:
        obs, _ = env.reset(seed=seed)
        # The collector needs a live connection, so it is built after reset and then
        # attached as the per-second hook for the rest of the episode.
        junction_lanes = list(env.traffic_signals[CORRIDOR_TLS_ID].lanes)
        collector = MetricsCollector(env.sumo, junction_lanes=junction_lanes)
        env.set_step_hook(lambda _conn: collector.sample())

        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
    finally:
        env.close()

    controller = controller_label or (f"ppo_{tag}" if tag else "ppo")
    return _rows_from(collector, controller, seed)


def run(controller: str, seeds: list[int], tag: str = "", csv_path: str = WEEK3_CSV) -> None:
    """Evaluate one controller across seeds and append the results to a CSV.

    Args:
        controller: one of :data:`CONTROLLERS`.
        seeds: SUMO seeds to evaluate.
        tag: optional PPO model tag.
        csv_path: destination CSV. Give each concurrently-running process its own
            file and merge afterwards — appending to one file from several processes
            can interleave partial lines.
    """
    for seed in seeds:
        if controller == "ppo":
            rows = eval_ppo(seed, tag)
        else:
            rows = eval_baseline(controller, seed)
        _save_rows(rows, csv_path)
        for row in rows:
            log.info(
                "%-16s seed=%d scope=%-8s wait=%7.2fs queue=%4s throughput=%s",
                row["controller"], seed, row["scope"],
                row["avg_wait_time_s"], row["max_queue_len"], row["throughput_veh"],
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Week 3 corridor controllers.")
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--tag", type=str, default="", help="Optional PPO model tag, e.g. 'short'.")
    parser.add_argument("--out", type=str, default=WEEK3_CSV, help="Destination CSV path.")
    args = parser.parse_args()
    run(args.controller, args.seeds, args.tag, args.out)


if __name__ == "__main__":
    main()
