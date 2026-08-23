"""Train multi-agent PPO on the SmartFlow corridor with Ray RLlib.

Two modes, matching the roadmap:

``--mode independent`` (Week 4)
    One policy per junction, ``policy_mapping_fn`` = identity. Independent Learners:
    each agent treats the other eleven as part of the environment, which makes that
    environment non-stationary. Week 4's bar is that this trains end-to-end without
    diverging.

``--mode shared`` (Week 5)
    A single parameter-shared policy driving all twelve junctions. Every junction's
    experience updates the same weights, which is why the field-aligned observation
    layout in ``marl_env.ObservationAligner`` matters — index *k* must mean the same
    thing at every junction.

Week 5 also enables the shaped reward (``--reward shaped``) and runs **dual ascent**
on the Lagrange multiplier: after each training iteration the mean constraint
violation is read off the env-runner metrics, ``lambda`` is updated, and the new value
is broadcast to every runner process. That is what makes the fairness cap an actual
constraint rather than a fixed penalty weight.

Usage:
    # Week 4 - independent policies, default reward
    python src/train_marl_corridor.py --mode independent --seed 0

    # Week 5 - shared policy, shaped reward, Lagrangian fairness
    python src/train_marl_corridor.py --mode shared --seed 0 --reward shaped --fairness

Produces:
    models/marl_{mode}_seed{N}[_tag]/           RLlib checkpoint
    outputs/week4_{mode}_seed{N}[_tag]_training.json   per-iteration history
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

from smartflow_env import require_sumo_home
from week4_config import (
    CORRIDOR_NET,
    CORRIDOR_ROUTE,
    CORRIDOR_TLS_IDS,
    FULL_TIMESTEPS,
    GNN_EMBED_DIM,
    GNN_HEADS,
    GNN_HIDDEN_DIM,
    MAX_NEIGHBORS,
    NUM_ENV_RUNNERS,
    PPO_CONFIG,
    SIM_SECONDS,
    SRC_DIR,
    STEPS_PER_EPISODE,
    TRAIN_SEEDS,
    checkpoint_dir,
    training_log_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

ENV_NAME = "smartflow_corridor"

# Dual-ascent settings for the Lagrangian fairness constraint.
DUAL_STEP_SIZE = 0.05   # per training iteration, applied to the relative violation
DUAL_LAMBDA_MAX = 2.0   # penalty stays on the same scale as the delay reward
DUAL_LAMBDA_INIT = 0.5


def _limit_threads(threads: int = 1) -> None:
    """Cap BLAS/torch thread fan-out in this process.

    The workload is many small processes each driving one SUMO simulation, not one
    process doing large matrix work. Left at their defaults, torch and the BLAS
    backends each spin up one thread per core inside every worker, which oversubscribes
    the machine badly once more than one training job is running.

    Args:
        threads: threads per process.
    """
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception as exc:  # pragma: no cover
        log.warning("Could not set torch thread count: %s", exc)


def _unwrap_corridor(env: Any) -> Any | None:
    """Dig the :class:`marl_env.CorridorParallelEnv` out of RLlib's wrappers.

    RLlib hands callbacks whatever wrapper it happens to be holding, which differs
    between vectorised and non-vectorised runners. Rather than assume one shape, walk
    the common attributes until something exposes ``fairness_summary``.

    Args:
        env: whatever RLlib passed to the callback.

    Returns:
        The corridor environment, or ``None`` if it could not be located.
    """
    seen = set()
    stack = [env]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "fairness_summary"):
            return candidate
        for attr in ("par_env", "unwrapped", "env", "envs"):
            child = getattr(candidate, attr, None)
            if isinstance(child, (list, tuple)):
                stack.extend(child)
            elif child is not None and child is not candidate:
                stack.append(child)
    return None


def _make_callback_class() -> type:
    """Build the fairness-reporting callback bound to the installed RLlib.

    Returns:
        An ``RLlibCallback`` subclass that logs per-episode fairness telemetry so the
        driver can run dual ascent on it.
    """
    from ray.rllib.callbacks.callbacks import RLlibCallback

    class FairnessCallback(RLlibCallback):
        """Log each episode's fairness statistics into the env-runner metrics."""

        def on_episode_end(self, *, episode: Any = None, env: Any = None,
                           metrics_logger: Any = None, **kwargs: Any) -> None:
            """Record max lane wait and constraint violation for the finished episode."""
            if metrics_logger is None:
                return
            corridor = _unwrap_corridor(env)
            if corridor is None:
                return
            for key, value in corridor.fairness_summary().items():
                metrics_logger.log_value(key, float(value), reduce="mean")

    return FairnessCallback


def build_algorithm(
    mode: str,
    seed: int,
    *,
    reward: str,
    num_env_runners: int,
    num_seconds: int,
    route_file: str,
    net_file: str,
    include_agent_id: bool,
    use_gnn: bool,
    reward_weights: dict[str, float] | None = None,
) -> Any:
    """Configure and build the RLlib PPO algorithm.

    Args:
        mode: ``"independent"`` (policy per junction) or ``"shared"`` (one policy).
        seed: RNG seed for RLlib and SUMO.
        reward: reward name resolved by ``rewards.make_reward_fn``.
        num_env_runners: parallel SUMO sampling processes.
        num_seconds: simulated seconds per episode.
        route_file: SUMO route file (Week 6 varies this to change demand).
        net_file: SUMO network file.
        include_agent_id: append a one-hot junction identity to observations.
        use_gnn: use the graph-attention encoder instead of the default MLP.
        reward_weights: overrides for ``rewards.RewardWeights`` fields, e.g.
            ``{"fairness_scale": 0.5}``.

    Returns:
        A built RLlib ``Algorithm``.

    Raises:
        ValueError: if ``mode`` is unknown.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from marl_env import rllib_env_creator

    register_env(ENV_NAME, rllib_env_creator)

    if mode == "independent":
        policies = set(CORRIDOR_TLS_IDS)

        def policy_mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
            return agent_id

    elif mode == "shared":
        policies = {"shared_policy"}

        def policy_mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
            return "shared_policy"

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'independent' or 'shared'.")

    env_config = {
        "net_file": net_file,
        "route_file": route_file,
        "controlled_ts": CORRIDOR_TLS_IDS,
        "num_seconds": num_seconds,
        "seed": seed,
        "reward_fn": reward,
        "include_agent_id": include_agent_id,
        # The GAT torso reads neighbour features out of the observation, so the env
        # must ship them; with the default MLP torso they would just be dead inputs.
        "neighbor_context": use_gnn,
        "max_neighbors": MAX_NEIGHBORS,
        "reward_weights": dict(reward_weights or {}),
    }

    # One full episode per runner per iteration keeps batches aligned with episode
    # boundaries, which makes the reward curve readable.
    train_batch_size = max(1, num_env_runners) * STEPS_PER_EPISODE

    # RLlib rejects a minibatch larger than the train batch. With few runners the
    # train batch shrinks below the configured minibatch, so clamp rather than let a
    # low-runner run (the online-learning loop, a quick smoke test) fail at build time.
    training_config = dict(PPO_CONFIG)
    configured_minibatch = int(training_config.get("minibatch_size", train_batch_size))
    if configured_minibatch > train_batch_size:
        log.warning(
            "minibatch_size %d exceeds train_batch_size %d (%d runners x %d steps); "
            "clamping to %d.",
            configured_minibatch, train_batch_size, num_env_runners,
            STEPS_PER_EPISODE, train_batch_size,
        )
        training_config["minibatch_size"] = train_batch_size

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(train_batch_size=train_batch_size, **training_config)
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=STEPS_PER_EPISODE,
            sample_timeout_s=1800.0,
        )
        .callbacks(_make_callback_class())
        # Every env runner owns a SUMO subprocess, and launching many at once can make
        # one lose its TraCI connection during construction ("Connection closed by
        # SUMO"). Without this, that single transient failure takes down the whole
        # training job. Restarting the runner costs one episode of samples instead.
        .fault_tolerance(
            restart_failed_env_runners=True,
            ignore_env_runner_failures=False,
            max_num_env_runner_restarts=10,
            delay_between_env_runner_restarts_s=10.0,
            num_consecutive_env_runner_failures_tolerance=5,
        )
        .debugging(seed=seed, log_level="ERROR")
    )

    if use_gnn:
        from gnn_encoder import build_gnn_module_spec
        from marl_env import make_parallel_env

        # The per-junction feature width depends on the corridor (max phases, max
        # lanes, optional agent-id one-hot), so read it off a throwaway env rather
        # than hard-coding it.
        probe = make_parallel_env({**env_config, "num_seconds": 60})
        try:
            node_dim = probe.node_dim
        finally:
            probe.close()
        config = config.rl_module(
            rl_module_spec=build_gnn_module_spec(
                policies,
                node_dim=node_dim,
                max_neighbors=MAX_NEIGHBORS,
                hidden_dim=GNN_HIDDEN_DIM,
                embed_dim=GNN_EMBED_DIM,
                heads=GNN_HEADS,
            )
        )

    return config.build_algo()


def train(
    mode: str,
    seed: int,
    timesteps: int,
    *,
    reward: str = "diff-waiting-time",
    fairness: bool = False,
    tag: str = "",
    num_env_runners: int = NUM_ENV_RUNNERS,
    num_seconds: int = SIM_SECONDS,
    route_file: str = CORRIDOR_ROUTE,
    net_file: str = CORRIDOR_NET,
    include_agent_id: bool = False,
    use_gnn: bool = False,
    reward_weights: dict[str, float] | None = None,
) -> str:
    """Run one MARL training job and save its checkpoint and history.

    Args:
        mode: ``"independent"`` or ``"shared"``.
        seed: training seed.
        timesteps: budget in *environment* steps (12 agent-steps each).
        reward: reward-function name.
        fairness: run dual ascent on the Lagrange multiplier between iterations.
        tag: optional run label.
        num_env_runners: parallel SUMO sampling processes.
        num_seconds: simulated seconds per episode.
        route_file: SUMO route file.
        net_file: SUMO network file.
        include_agent_id: append one-hot junction identity to observations.
        use_gnn: use the graph-attention encoder.
        reward_weights: overrides for ``rewards.RewardWeights`` fields.

    Returns:
        Path to the saved RLlib checkpoint.
    """
    require_sumo_home()
    _limit_threads()
    import ray

    from rewards import dual_ascent, RewardWeights

    # Remote env runners are separate processes that must be able to import this
    # project's modules and reach SUMO; PYTHONPATH + SUMO_HOME are the only things
    # they need, so no working_dir upload is required.
    #
    # num_cpus is pinned deliberately. Ray otherwise sizes each cluster to the whole
    # machine, so running several training jobs side by side spawns hundreds of
    # worker processes competing for the same cores and everything crawls. One CPU
    # per env runner plus one for the driver/learner is what this job actually uses.
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=max(2, num_env_runners + 1),
        runtime_env={
            "env_vars": {
                "PYTHONPATH": SRC_DIR,
                "SUMO_HOME": os.environ["SUMO_HOME"],
                # Each runner is a single SUMO simulation; letting torch/BLAS fan out
                # to every core inside every worker is pure contention.
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        },
    )

    algo = build_algorithm(
        mode, seed,
        reward=reward,
        num_env_runners=num_env_runners,
        num_seconds=num_seconds,
        route_file=route_file,
        net_file=net_file,
        include_agent_id=include_agent_id,
        use_gnn=use_gnn,
        reward_weights=reward_weights,
    )

    weights = RewardWeights(**(reward_weights or {}))
    cap = weights.max_wait_s
    lam = DUAL_LAMBDA_INIT
    history: list[dict[str, Any]] = []
    total_steps = 0
    started = time.perf_counter()

    try:
        if fairness:
            _broadcast_lambda(algo, lam)

        while total_steps < timesteps:
            result = algo.train()
            runners = result.get("env_runners", {})
            total_steps = int(result.get("num_env_steps_sampled_lifetime", total_steps))
            reward_mean = runners.get("episode_return_mean")
            max_lane_wait = runners.get("max_lane_wait_s")
            violation = runners.get("mean_violation_s")

            entry = {
                "iteration": len(history) + 1,
                "env_steps": total_steps,
                "episode_return_mean": _as_float(reward_mean),
                "max_lane_wait_s": _as_float(max_lane_wait),
                "mean_violation_s": _as_float(violation),
                "fairness_lambda": lam,
                "elapsed_s": round(time.perf_counter() - started, 1),
            }

            if fairness and max_lane_wait is not None:
                # Dual ascent on the constraint "max lane wait <= cap", using the
                # relative gap so the update scale does not depend on how many
                # seconds of violation the corridor happens to produce.
                violation_ratio = (float(max_lane_wait) - cap) / cap
                lam = dual_ascent(lam, violation_ratio, DUAL_STEP_SIZE, DUAL_LAMBDA_MAX)
                _broadcast_lambda(algo, lam)
                entry["fairness_lambda_next"] = lam

            history.append(entry)
            log.info(
                "%s seed=%d iter=%d steps=%d/%d return=%s max_lane_wait=%s lambda=%.3f",
                mode, seed, entry["iteration"], total_steps, timesteps,
                _fmt(reward_mean), _fmt(max_lane_wait), lam,
            )

        out_dir = checkpoint_dir(mode, seed, tag)
        checkpoint = algo.save(out_dir)
        duration = time.perf_counter() - started

        metadata = {
            "mode": mode,
            "seed": seed,
            "tag": tag or None,
            "reward_fn": reward,
            "fairness_dual_ascent": fairness,
            "fairness_cap_s": cap,
            "fairness_scale": weights.fairness_scale,
            "coordination_alpha": weights.coordination_alpha,
            "fairness_lambda_final": lam,
            "dual_step_size": DUAL_STEP_SIZE if fairness else None,
            "timesteps_requested": timesteps,
            "timesteps_actual": total_steps,
            "agent_steps_actual": total_steps * len(CORRIDOR_TLS_IDS),
            "num_env_runners": num_env_runners,
            "sim_seconds": num_seconds,
            "net_file": net_file,
            "route_file": route_file,
            "include_agent_id": include_agent_id,
            "use_gnn": use_gnn,
            "ppo_config": PPO_CONFIG,
            "duration_s": round(duration, 1),
            "env_steps_per_s": round(total_steps / max(duration, 1e-6), 1),
            "checkpoint": str(checkpoint),
            "iterations": history,
        }
        path = training_log_path(mode, seed, tag)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        log.info("Saved checkpoint -> %s", out_dir)
        log.info("Saved history    -> %s", path)
        return out_dir
    finally:
        algo.stop()
        ray.shutdown()


def _broadcast_lambda(algo: Any, lam: float) -> None:
    """Push a new Lagrange multiplier into every env-runner process.

    Each runner is a separate process with its own import of ``rewards``, so the
    multiplier has to be set in all of them, not just the driver.

    Args:
        algo: the RLlib algorithm.
        lam: the new multiplier.
    """

    def _setter(_runner: Any) -> None:
        import rewards

        rewards.set_lambda(lam)

    try:
        algo.env_runner_group.foreach_env_runner(_setter, local_env_runner=True)
    except Exception as exc:  # pragma: no cover - depends on runner health
        log.warning("Could not broadcast fairness lambda=%.3f to env runners: %s", lam, exc)


def _as_float(value: Any) -> float | None:
    """Coerce an RLlib metric to ``float``, tolerating ``None`` and NaN."""
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    """Format an optional metric for log output."""
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train RLlib multi-agent PPO on the SmartFlow corridor.")
    parser.add_argument("--mode", choices=["independent", "shared"], default="independent")
    parser.add_argument("--seed", type=int, default=TRAIN_SEEDS[0])
    parser.add_argument("--timesteps", type=int, default=FULL_TIMESTEPS)
    parser.add_argument("--reward", type=str, default="diff-waiting-time")
    parser.add_argument("--fairness", action="store_true", help="Run dual ascent on the fairness multiplier.")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--num-env-runners", type=int, default=NUM_ENV_RUNNERS)
    parser.add_argument("--num-seconds", type=int, default=SIM_SECONDS)
    parser.add_argument("--route-file", type=str, default=CORRIDOR_ROUTE)
    parser.add_argument("--net-file", type=str, default=CORRIDOR_NET)
    parser.add_argument("--include-agent-id", action="store_true")
    parser.add_argument("--gnn", action="store_true", help="Use the graph-attention state encoder.")
    parser.add_argument("--fairness-scale", type=float, default=None,
                        help="Override rewards.RewardWeights.fairness_scale.")
    parser.add_argument("--max-wait-s", type=float, default=None,
                        help="Override the per-vehicle fairness cap in seconds.")
    args = parser.parse_args()

    overrides: dict[str, float] = {}
    if args.fairness_scale is not None:
        overrides["fairness_scale"] = args.fairness_scale
    if args.max_wait_s is not None:
        overrides["max_wait_s"] = args.max_wait_s

    train(
        args.mode, args.seed, args.timesteps,
        reward=args.reward,
        fairness=args.fairness,
        tag=args.tag,
        num_env_runners=args.num_env_runners,
        num_seconds=args.num_seconds,
        route_file=args.route_file,
        net_file=args.net_file,
        include_agent_id=args.include_agent_id,
        use_gnn=args.gnn,
        reward_weights=overrides or None,
    )


if __name__ == "__main__":
    main()
