"""Week 5 online learning loop: keep adapting after deployment.

A policy trained offline on one demand pattern is deployed into a *different* one and
keeps learning from the live stream. This is the deployment-shaped question the
roadmap's "online learning loop" asks: not "can we train a policy?" but "when the
traffic changes under a policy that is already running, does it recover?"

The loop:

1. Build a shared-policy PPO algorithm pointed at the **shifted** demand scenario.
2. Restore the weights trained on the **base** demand (offline phase).
3. Round 0 — evaluate the restored policy deterministically. This is the frozen
   baseline: what you get if the deployed policy never updates.
4. Rounds 1..N — one PPO update per round on freshly collected experience, then a
   deterministic evaluation.

Because the frozen policy is deterministic and the demand is fixed, its score is a
constant line, so any movement in the online curve is adaptation rather than noise.

Usage:
    python src/online_learning.py --seed 0 --tag w5 --scenario asymmetric --rounds 10

Produces:
    outputs/week5_online_learning_{scenario}.json   per-round metrics
    outputs/week5_online_learning_{scenario}.png    adaptation curve
    models/marl_shared_seed{N}_online/              the adapted policy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import reward_curve_chart
from eval_marl_corridor import load_multi_rl_module, rollout_module
from scenarios import resolve_scenario
from smartflow_env import require_sumo_home
from train_marl_corridor import build_algorithm
from week4_config import MODELS_DIR, OUTPUTS_DIR, SIM_SECONDS, SRC_DIR, checkpoint_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# Restoring only this subcomponent leaves the algorithm's env config, optimiser and
# step counters as configured for the *new* demand.
RL_MODULE_COMPONENT = "learner_group/learner/rl_module"


def online_json_path(scenario: str) -> str:
    """Return the results JSON path for one scenario's online run."""
    return os.path.join(OUTPUTS_DIR, f"week5_online_learning_{scenario}.json")


def online_png_path(scenario: str) -> str:
    """Return the adaptation-curve PNG path for one scenario's online run."""
    return os.path.join(OUTPUTS_DIR, f"week5_online_learning_{scenario}.png")


def _restore_policy(algo: Any, source_checkpoint: str) -> None:
    """Load only the policy weights from an existing checkpoint into a fresh algorithm.

    The algorithm was configured for the *shifted* demand, so only the RLModule state
    is restored — env config, optimiser settings and step counters stay as configured.

    Args:
        algo: the freshly built RLlib algorithm.
        source_checkpoint: path to the offline-trained algorithm checkpoint.

    Raises:
        FileNotFoundError: if the source checkpoint has no ``rl_module`` directory.
    """
    module_dir = os.path.join(source_checkpoint, "learner_group", "learner", "rl_module")
    if not os.path.isdir(module_dir):
        raise FileNotFoundError(
            f"No rl_module inside {source_checkpoint}. "
            "Train the offline Week 5 policy first with train_marl_corridor.py --mode shared."
        )
    algo.restore_from_path(module_dir, component=RL_MODULE_COMPONENT)
    # Push the restored weights out to the sampling processes before the first round.
    try:
        algo.env_runner_group.sync_weights()
    except Exception as exc:  # pragma: no cover - depends on runner health
        log.warning("Could not sync restored weights to env runners: %s", exc)
    log.info("Restored offline policy from %s", module_dir)


def _evaluate(algo: Any, seed: int, net_file: str, route_file: str, scenario: str,
              reward: str, num_seconds: int, scratch_dir: str) -> tuple[dict[str, Any], dict[str, float]]:
    """Checkpoint the current policy and score it on one deterministic episode.

    Args:
        algo: the running algorithm.
        seed: SUMO seed for the evaluation episode.
        net_file: SUMO network file.
        route_file: SUMO route file for the shifted demand.
        scenario: scenario label.
        reward: reward name (drives fairness telemetry).
        num_seconds: episode length.
        scratch_dir: directory reused for the throwaway evaluation checkpoint.

    Returns:
        ``(corridor metrics row, fairness summary)``.
    """
    algo.save(scratch_dir)
    module = load_multi_rl_module(scratch_dir)
    rows, fairness = rollout_module(
        module, "shared", seed, net_file=net_file, route_file=route_file,
        scenario=scenario, num_seconds=num_seconds, reward=reward,
        controller_label="marl_shared_online",
    )
    corridor = next(row for row in rows if row["scope"] == "corridor")
    return corridor, fairness


def run(
    seed: int,
    *,
    tag: str = "",
    scenario: str = "asymmetric",
    rounds: int = 10,
    reward: str = "shaped",
    num_env_runners: int = 4,
    num_seconds: int = SIM_SECONDS,
) -> str:
    """Run the online learning loop and write its results.

    Args:
        seed: seed of the offline-trained policy to start from, and the eval seed.
        tag: tag of the offline checkpoint (e.g. ``"w5"``).
        scenario: demand scenario to deploy into.
        rounds: number of online update rounds.
        reward: reward name used online.
        num_env_runners: parallel SUMO sampling processes.
        num_seconds: simulated seconds per episode.

    Returns:
        Path to the results JSON.
    """
    require_sumo_home()
    import ray

    net_file, route_file = resolve_scenario(scenario)
    source_checkpoint = checkpoint_dir("shared", seed, tag)

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": SRC_DIR, "SUMO_HOME": os.environ["SUMO_HOME"]}},
    )

    algo = build_algorithm(
        "shared", seed,
        reward=reward,
        num_env_runners=num_env_runners,
        num_seconds=num_seconds,
        route_file=route_file,
        net_file=net_file,
        include_agent_id=False,
        use_gnn=False,
    )

    scratch_dir = os.path.join(MODELS_DIR, f"_online_scratch_seed{seed}")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        _restore_policy(algo, source_checkpoint)

        # Round 0: the frozen, never-updated policy in the new demand.
        frozen, frozen_fairness = _evaluate(
            algo, seed, net_file, route_file, scenario, reward, num_seconds, scratch_dir
        )
        history.append({
            "round": 0, "updated": False,
            "avg_wait_time_s": frozen["avg_wait_time_s"],
            "throughput_veh": frozen["throughput_veh"],
            "max_queue_len": frozen["max_queue_len"],
            "wait_p95_s": frozen.get("wait_p95_s"),
            "max_lane_wait_s": frozen_fairness.get("max_lane_wait_s"),
            "elapsed_s": round(time.perf_counter() - started, 1),
        })
        log.info(
            "round 0 (frozen offline policy on '%s'): wait=%.2fs throughput=%s",
            scenario, frozen["avg_wait_time_s"], frozen["throughput_veh"],
        )

        for index in range(1, rounds + 1):
            algo.train()
            corridor, fairness = _evaluate(
                algo, seed, net_file, route_file, scenario, reward, num_seconds, scratch_dir
            )
            history.append({
                "round": index, "updated": True,
                "avg_wait_time_s": corridor["avg_wait_time_s"],
                "throughput_veh": corridor["throughput_veh"],
                "max_queue_len": corridor["max_queue_len"],
                "wait_p95_s": corridor.get("wait_p95_s"),
                "max_lane_wait_s": fairness.get("max_lane_wait_s"),
                "elapsed_s": round(time.perf_counter() - started, 1),
            })
            log.info(
                "round %d: wait=%.2fs throughput=%s max_queue=%s",
                index, corridor["avg_wait_time_s"], corridor["throughput_veh"], corridor["max_queue_len"],
            )

        adapted_dir = checkpoint_dir("shared", seed, "online")
        algo.save(adapted_dir)

        frozen_wait = history[0]["avg_wait_time_s"]
        best_wait = min(entry["avg_wait_time_s"] for entry in history)
        final_wait = history[-1]["avg_wait_time_s"]
        result = {
            "seed": seed,
            "source_checkpoint": source_checkpoint,
            "adapted_checkpoint": adapted_dir,
            "scenario": scenario,
            "reward_fn": reward,
            "rounds": rounds,
            "num_env_runners": num_env_runners,
            "sim_seconds": num_seconds,
            "frozen_avg_wait_s": frozen_wait,
            "final_avg_wait_s": final_wait,
            "best_avg_wait_s": best_wait,
            "improvement_pct_final": round((frozen_wait - final_wait) / frozen_wait * 100, 2) if frozen_wait else 0.0,
            "improvement_pct_best": round((frozen_wait - best_wait) / frozen_wait * 100, 2) if frozen_wait else 0.0,
            "duration_s": round(time.perf_counter() - started, 1),
            "history": history,
        }
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        json_path = online_json_path(scenario)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        log.info("Wrote %s", json_path)

        rounds_axis = [float(entry["round"]) for entry in history]
        reward_curve_chart(
            {
                "online (keeps learning)": (rounds_axis, [entry["avg_wait_time_s"] for entry in history]),
                "frozen offline policy": (rounds_axis, [frozen_wait] * len(history)),
            },
            online_png_path(scenario),
            title=f"Week 5 — online adaptation to '{scenario}' demand (seed {seed})",
            xlabel="Online update round",
            ylabel="Corridor avg wait (s)",
        )
        return json_path
    finally:
        algo.stop()
        ray.shutdown()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Week 5 online learning loop.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="w5", help="Tag of the offline checkpoint.")
    parser.add_argument("--scenario", type=str, default="asymmetric")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--reward", type=str, default="shaped")
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--num-seconds", type=int, default=SIM_SECONDS)
    args = parser.parse_args()

    run(
        args.seed, tag=args.tag, scenario=args.scenario, rounds=args.rounds,
        reward=args.reward, num_env_runners=args.num_env_runners, num_seconds=args.num_seconds,
    )


if __name__ == "__main__":
    main()
