"""Evaluation harness for corridor-wide controllers (Weeks 4-6).

Evaluates a trained RLlib multi-agent checkpoint — or a non-RL baseline — on the
corridor and logs the same metrics the Week 1/3 harness produces, so every number in
the project remains directly comparable.

Actions are taken **deterministically**: the policy's action-distribution logits are
argmax-ed rather than sampled. Reporting a sampled rollout would mix policy quality
with exploration noise.

Beyond the four headline metrics this harness also records two fairness statistics,
which is what Week 5's "fairness constraint measurably caps worst-case wait" claim is
judged on:

``wait_p95_s``
    95th percentile of completed-trip waiting time — the tail, not the average.
``worst_vehicle_wait_s``
    The single worst trip in the episode.

Usage:
    python src/eval_marl_corridor.py --controller fixed --seeds 0 1 2
    python src/eval_marl_corridor.py --controller marl --mode independent --seeds 0 1 2
    python src/eval_marl_corridor.py --controller marl --mode shared --tag w5 --seeds 0 1 2
    python src/eval_marl_corridor.py --controller marl --mode shared --tag w5 --scenario peak
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
from smartflow_env import baseline_sumo_cmd, require_sumo_home
from week3_config import CORRIDOR_TLS_ID
from week4_config import CORRIDOR_NET, CORRIDOR_ROUTE, CORRIDOR_TLS_IDS, OUTPUTS_DIR, SIM_SECONDS, TRAIN_SEEDS, checkpoint_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

MARL_CSV = os.path.join(OUTPUTS_DIR, "marl_metrics.csv")

CSV_HEADER = [
    "controller",
    "seed",
    "scenario",
    "scope",
    "tls_id",
    "avg_wait_time_s",
    "max_queue_len",
    "throughput_veh",
    "total_co2_kg",
    "wait_p95_s",
    "worst_vehicle_wait_s",
]


def save_rows(rows: list[dict[str, Any]], csv_path: str = MARL_CSV) -> None:
    """Append metric rows to a CSV, writing the header on first use.

    Args:
        rows: dicts whose keys are a subset of :data:`CSV_HEADER`.
        csv_path: destination CSV. Give concurrent processes separate files.
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
        raise OSError(f"Could not append metrics to {csv_path}: {exc}") from exc


def _rows_from(collector: MetricsCollector, controller: str, seed: int, scenario: str) -> list[dict[str, Any]]:
    """Build the junction-scope and corridor-scope rows for one finished episode."""
    corridor = collector.corridor_row()
    junction = collector.junction_row()
    tail = {
        "wait_p95_s": collector.wait_percentile(95),
        "worst_vehicle_wait_s": collector.max_lane_wait_s(),
    }
    return [
        {
            "controller": controller, "seed": seed, "scenario": scenario,
            "scope": "junction", "tls_id": CORRIDOR_TLS_ID, "total_co2_kg": "",
            **junction,
        },
        {
            "controller": controller, "seed": seed, "scenario": scenario,
            "scope": "corridor", "tls_id": "all",
            **corridor, **tail,
        },
    ]


def eval_baseline(
    controller: str, seed: int, *, net_file: str, route_file: str,
    scenario: str = "base", num_seconds: int = SIM_SECONDS,
) -> list[dict[str, Any]]:
    """Run a fixed-time or actuated baseline for one episode.

    Args:
        controller: label to record, e.g. ``"fixed"`` or ``"actuated"``.
        seed: SUMO RNG seed.
        net_file: SUMO network (choose the actuated net for the actuated baseline).
        route_file: SUMO route file.
        scenario: demand-scenario label recorded in the CSV.
        num_seconds: episode length.

    Returns:
        Junction-scope and corridor-scope rows.
    """
    require_sumo_home()
    import traci

    for label, path in (("net_file", net_file), ("route_file", route_file)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    traci.start(baseline_sumo_cmd(net_file=net_file, route_file=route_file, seed=seed))
    try:
        junction_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(CORRIDOR_TLS_ID)))
        collector = MetricsCollector(traci, junction_lanes=junction_lanes)
        while traci.simulation.getTime() < num_seconds:
            traci.simulationStep()
            collector.sample()
    finally:
        traci.close()
    return _rows_from(collector, controller, seed, scenario)


def load_multi_rl_module(checkpoint: str) -> Any:
    """Load the ``MultiRLModule`` saved inside an RLlib algorithm checkpoint.

    Args:
        checkpoint: path to the algorithm checkpoint directory.

    Returns:
        The loaded ``MultiRLModule``.

    Raises:
        FileNotFoundError: if the checkpoint has no ``rl_module`` subdirectory.
    """
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

    module_dir = os.path.join(checkpoint, "learner_group", "learner", "rl_module")
    if not os.path.isdir(module_dir):
        raise FileNotFoundError(
            f"No rl_module found under {checkpoint}. "
            "Expected an RLlib algorithm checkpoint produced by train_marl_corridor.py."
        )
    return MultiRLModule.from_checkpoint(module_dir)


def rollout_module(
    module: Any, mode: str, seed: int, *, net_file: str = CORRIDOR_NET,
    route_file: str = CORRIDOR_ROUTE, scenario: str = "base",
    num_seconds: int = SIM_SECONDS, reward: str = "diff-waiting-time",
    use_gnn: bool = False, controller_label: str = "marl",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run one deterministic episode with an already-loaded ``MultiRLModule``.

    Split out from :func:`eval_marl` so the Week 5 online-learning loop can evaluate
    a policy that is still being updated, without going through disk each round.

    Args:
        module: a loaded ``MultiRLModule``.
        mode: ``"independent"`` or ``"shared"``, selecting the policy mapping.
        seed: SUMO RNG seed.
        net_file: SUMO network file.
        route_file: SUMO route file.
        scenario: demand-scenario label recorded in the CSV.
        num_seconds: episode length.
        reward: reward name used to build the env (drives fairness telemetry).
        use_gnn: whether the policy expects neighbour context in observations.
        controller_label: value written to the CSV's ``controller`` column.

    Returns:
        ``(rows, fairness_summary)``.
    """
    require_sumo_home()
    import torch
    from ray.rllib.core.columns import Columns

    from marl_env import make_parallel_env

    policy_for = (lambda agent: agent) if mode == "independent" else (lambda agent: "shared_policy")

    env = make_parallel_env({
        "net_file": net_file,
        "route_file": route_file,
        "controlled_ts": CORRIDOR_TLS_IDS,
        "num_seconds": num_seconds,
        "seed": seed,
        "reward_fn": reward,
        "neighbor_context": use_gnn,
    })
    try:
        obs, _ = env.reset(seed=seed)
        junction_lanes = list(env.traffic_signals[CORRIDOR_TLS_ID].lanes)
        collector = MetricsCollector(env.sumo, junction_lanes=junction_lanes)
        env.set_step_hook(lambda _conn: collector.sample())

        while env.agents:
            actions: dict[str, int] = {}
            with torch.no_grad():
                for agent, agent_obs in obs.items():
                    rl_module = module[policy_for(agent)]
                    batch = {Columns.OBS: torch.as_tensor(agent_obs, dtype=torch.float32).unsqueeze(0)}
                    out = rl_module.forward_inference(batch)
                    # Deterministic action: argmax of the policy logits, never a sample.
                    actions[agent] = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
            obs, _rewards, _term, _trunc, _infos = env.step(actions)
        fairness = env.fairness_summary()
    finally:
        env.close()

    return _rows_from(collector, controller_label, seed, scenario), fairness


def eval_marl(
    mode: str, seed: int, *, tag: str = "", net_file: str = CORRIDOR_NET,
    route_file: str = CORRIDOR_ROUTE, scenario: str = "base",
    num_seconds: int = SIM_SECONDS, reward: str = "diff-waiting-time",
    use_gnn: bool = False, controller_label: str | None = None,
) -> list[dict[str, Any]]:
    """Run a trained multi-agent policy deterministically for one episode.

    Args:
        mode: ``"independent"`` or ``"shared"`` (selects the policy mapping).
        seed: SUMO seed and checkpoint selector.
        tag: optional checkpoint tag.
        net_file: SUMO network file.
        route_file: SUMO route file.
        scenario: demand-scenario label recorded in the CSV.
        num_seconds: episode length.
        reward: reward name the env was trained with (affects nothing at eval time
            except the fairness telemetry, which needs the shaped reward to populate).
        use_gnn: whether the checkpoint uses the graph-attention encoder, which
            requires neighbour context in the observation.
        controller_label: override the ``controller`` value written to the CSV.

    Returns:
        Junction-scope and corridor-scope rows.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
    """
    checkpoint = checkpoint_dir(mode, seed, tag)
    if not os.path.isdir(os.path.join(checkpoint, "learner_group")):
        raise FileNotFoundError(
            f"No trained checkpoint at {checkpoint}. Train it first with "
            f"'python src/train_marl_corridor.py --mode {mode} --seed {seed}"
            + (f" --tag {tag}'" if tag else "'")
        )

    module = load_multi_rl_module(checkpoint)
    label = controller_label or f"marl_{mode}" + (f"_{tag}" if tag else "")
    rows, fairness = rollout_module(
        module, mode, seed, net_file=net_file, route_file=route_file, scenario=scenario,
        num_seconds=num_seconds, reward=reward, use_gnn=use_gnn, controller_label=label,
    )
    log.info("%s seed=%d scenario=%s episode fairness telemetry: %s", label, seed, scenario, fairness)
    return rows


def main() -> None:
    """CLI entry point."""
    from scenarios import SCENARIOS, resolve_scenario

    parser = argparse.ArgumentParser(description="Evaluate corridor-wide controllers.")
    parser.add_argument("--controller", choices=["fixed", "actuated", "marl"], required=True)
    parser.add_argument("--mode", choices=["independent", "shared"], default="shared")
    parser.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="base")
    parser.add_argument("--reward", type=str, default="diff-waiting-time")
    parser.add_argument("--gnn", action="store_true")
    parser.add_argument("--label", type=str, default="", help="Override the CSV controller label.")
    parser.add_argument("--out", type=str, default=MARL_CSV)
    args = parser.parse_args()

    net_file, route_file = resolve_scenario(args.scenario, actuated=(args.controller == "actuated"))

    for seed in args.seeds:
        if args.controller == "marl":
            rows = eval_marl(
                args.mode, seed, tag=args.tag, net_file=net_file, route_file=route_file,
                scenario=args.scenario, reward=args.reward, use_gnn=args.gnn,
                controller_label=args.label or None,
            )
        else:
            rows = eval_baseline(
                args.label or args.controller, seed,
                net_file=net_file, route_file=route_file, scenario=args.scenario,
            )
        save_rows(rows, args.out)
        for row in rows:
            log.info(
                "%-22s seed=%d scenario=%-10s scope=%-8s wait=%7.2fs queue=%4s throughput=%s",
                row["controller"], seed, args.scenario, row["scope"],
                row["avg_wait_time_s"], row["max_queue_len"], row["throughput_veh"],
            )


if __name__ == "__main__":
    main()
