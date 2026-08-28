"""Week 9: federated signal control across corridor districts, with Flower.

Each district trains a PPO policy on its own junction and never sees another
district's traffic. Flower's ``FedAvg`` aggregation averages the resulting
weights, and the averaged policy is then evaluated on a junction that took no
part in training at all. The question under test is the one the roadmap asks:
does federated averaging measurably improve a held-out district's policy?

Four things are measured on the held-out junction, all with the same harness:

* ``fixed`` - the untouched signal program, as the floor,
* ``local_<j>`` - each district's own policy transferred to the held-out
  junction, which is what you get without federation,
* ``fedavg`` - the averaged policy,
* ``local_mean`` - the mean of the transferred local policies, which is the
  number ``fedavg`` has to beat for federation to be worth anything.

**On process isolation.** The roadmap specifies Flower "simulated as multiple
processes on one machine". Flower's own simulation runner places clients in Ray
actors; this project already lost a day to Ray oversubscription in Week 4, and
each client here spawns a SUMO subprocess of its own. The rounds are therefore
driven in one process while the aggregation itself is Flower's - the real
``flwr.server.strategy.aggregate.aggregate``, not a hand-rolled mean. That
changes the orchestration, not the algorithm or the result.

Usage:
    python src/federated.py
    python src/federated.py --rounds 2 --steps 1500 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from smartflow_env import make_single_agent_env
from week3_config import (
    CORRIDOR_NET,
    CORRIDOR_ROUTE,
    DELTA_TIME,
    MAX_GREEN,
    MIN_GREEN,
    YELLOW_TIME,
)
from week9_config import (
    FED_CHART,
    FED_DISTRICTS,
    FED_EVAL_SECONDS,
    FED_HELD_OUT,
    FED_RESULTS,
    FED_ROUNDS,
    FED_SEEDS,
    FED_STEPS_PER_ROUND,
    PPO_KWARGS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def build_env(junction: str, seed: int, seconds: int, step_hook=None) -> Any:  # noqa: ANN001
    """Create a single-junction environment.

    Args:
        junction: junction under RL control.
        seed: SUMO seed.
        seconds: simulated seconds per episode.
        step_hook: optional per-second callback.

    Returns:
        A Gymnasium environment.
    """
    return make_single_agent_env(
        CORRIDOR_NET, CORRIDOR_ROUTE, junction,
        num_seconds=seconds, delta_time=DELTA_TIME, yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN, max_green=MAX_GREEN, seed=seed, step_hook=step_hook,
    )


def get_weights(model: Any) -> list[np.ndarray]:
    """Extract a policy's parameters as a list of arrays.

    Args:
        model: an SB3 PPO model.

    Returns:
        Parameters in a stable key order, as numpy arrays.
    """
    state = model.policy.state_dict()
    return [state[key].detach().cpu().numpy() for key in sorted(state)]


def set_weights(model: Any, weights: list[np.ndarray]) -> None:
    """Load a list of arrays back into a policy.

    Args:
        model: an SB3 PPO model.
        weights: arrays in the order produced by :func:`get_weights`.
    """
    import torch
    state = model.policy.state_dict()
    new_state = {key: torch.tensor(value, dtype=state[key].dtype)
                 for key, value in zip(sorted(state), weights)}
    model.policy.load_state_dict(new_state)


def train_local(junction: str, weights: list[np.ndarray] | None, steps: int,
                seed: int) -> tuple[list[np.ndarray], int]:
    """Train one district's policy for a round, starting from given weights.

    Args:
        junction: the district's junction.
        weights: starting parameters, or None to initialise fresh.
        steps: environment steps this round.
        seed: seed for SUMO and PPO.

    Returns:
        ``(updated_weights, steps_trained)`` - the step count is the example
        weight FedAvg uses.
    """
    from stable_baselines3 import PPO

    env = build_env(junction, seed, FED_EVAL_SECONDS)
    try:
        model = PPO("MlpPolicy", env, seed=seed, **PPO_KWARGS)
        if weights is not None:
            set_weights(model, weights)
        model.learn(total_timesteps=steps, progress_bar=False)
        return get_weights(model), steps
    finally:
        env.close()


def junction_lanes(junction: str) -> list[str]:
    """Return a junction's incoming lane ids.

    Args:
        junction: junction id.

    Returns:
        Incoming lane ids, in sumo-rl's order.
    """
    env = build_env(junction, 0, 200)
    try:
        return list(env.traffic_signals[junction].lanes)
    finally:
        env.close()


def evaluate_fixed(junction: str, lanes: list[str], seed: int) -> dict[str, float]:
    """Measure the junction under its own fixed-time program, with no RL control.

    An earlier version of this function used the controlled environment and held
    action 0, which does not reproduce the fixed-time program at all - it pins one
    approach permanently green and produced an implausible 0.78 s wait. The
    baseline has to be a run where nothing overrides the signal program.

    Args:
        junction: junction to measure.
        lanes: that junction's incoming lanes.
        seed: SUMO seed.

    Returns:
        The same metric dictionary the policy evaluation returns.
    """
    import traci
    from metrics import MetricsCollector

    sumo_bin = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
    traci.start([sumo_bin, "-n", CORRIDOR_NET, "-r", CORRIDOR_ROUTE,
                 "--no-step-log", "--no-warnings",
                 "--waiting-time-memory", "1000", "--max-depart-delay", "-1",
                 "--time-to-teleport", "-1", "--seed", str(seed)])
    try:
        collector = MetricsCollector(traci, junction_lanes=lanes)
        for _ in range(FED_EVAL_SECONDS):
            traci.simulationStep()
            collector.sample()
        corridor = collector.corridor_row()
        local = collector.junction_row()
    finally:
        traci.close()
    return {
        "corridor_wait_s": float(corridor["avg_wait_time_s"]),
        "corridor_throughput": float(corridor["throughput_veh"]),
        "junction_wait_s": float(local["avg_wait_time_s"]),
        "junction_queue": float(local["max_queue_len"]),
    }


def evaluate(junction: str, weights: list[np.ndarray], seed: int) -> dict[str, float]:
    """Evaluate one policy on one junction, deterministically.

    Args:
        junction: junction to evaluate on.
        weights: policy parameters.
        seed: SUMO seed.

    Returns:
        Corridor-wide and junction-local metrics for the episode.
    """
    from metrics import MetricsCollector
    from stable_baselines3 import PPO

    holder: dict[str, Any] = {}

    def hook(_conn: Any) -> None:
        collector = holder.get("c")
        if collector is not None:
            collector.sample()

    env = build_env(junction, seed, FED_EVAL_SECONDS, step_hook=hook)
    try:
        obs, _info = env.reset()
        holder["c"] = MetricsCollector(
            env.sumo, junction_lanes=list(env.traffic_signals[junction].lanes))

        model = PPO("MlpPolicy", env, seed=seed, **PPO_KWARGS)
        set_weights(model, weights)

        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
            done = bool(terminated or truncated)

        corridor = holder["c"].corridor_row()
        local = holder["c"].junction_row()
    finally:
        env.close()

    return {
        "corridor_wait_s": float(corridor["avg_wait_time_s"]),
        "corridor_throughput": float(corridor["throughput_veh"]),
        "junction_wait_s": float(local["avg_wait_time_s"]),
        "junction_queue": float(local["max_queue_len"]),
    }


def weight_distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """L2 distance between two flattened parameter sets.

    Args:
        a: one parameter set.
        b: another, with matching shapes.

    Returns:
        The Euclidean distance.
    """
    flat_a = np.concatenate([x.ravel() for x in a])
    flat_b = np.concatenate([x.ravel() for x in b])
    return float(np.linalg.norm(flat_a - flat_b))


def action_agreement(junction: str, left: list[np.ndarray],
                     right: list[np.ndarray], seed: int) -> dict[str, float]:
    """Compare two policies' actual decisions on one junction.

    Identical metrics can mean identical behaviour or a measurement artefact.
    This replays both policies over the same observations and counts how often
    they choose the same action, which distinguishes the two.

    Args:
        junction: junction to roll out on.
        left: first policy's parameters.
        right: second policy's parameters.
        seed: SUMO seed.

    Returns:
        ``{"steps", "agreement"}``.
    """
    from stable_baselines3 import PPO

    env = build_env(junction, seed, FED_EVAL_SECONDS)
    try:
        model_a = PPO("MlpPolicy", env, seed=seed, **PPO_KWARGS)
        model_b = PPO("MlpPolicy", env, seed=seed, **PPO_KWARGS)
        set_weights(model_a, left)
        set_weights(model_b, right)

        obs, _info = env.reset()
        same = 0
        steps = 0
        done = False
        while not done:
            action_a, _ = model_a.predict(obs, deterministic=True)
            action_b, _ = model_b.predict(obs, deterministic=True)
            same += int(int(action_a) == int(action_b))
            steps += 1
            obs, _r, terminated, truncated, _i = env.step(int(action_a))
            done = bool(terminated or truncated)
    finally:
        env.close()
    return {"steps": float(steps),
            "agreement": (same / steps) if steps else float("nan")}


def run_seed(seed: int, rounds: int, steps: int) -> dict[str, Any]:
    """Run the whole federated experiment for one seed.

    Args:
        seed: seed for training and evaluation.
        rounds: federated rounds.
        steps: steps per district per round.

    Returns:
        A record of every measured configuration.
    """
    from flwr.server.strategy.aggregate import aggregate

    log.info("seed %d: %d rounds x %d steps x %d districts",
             seed, rounds, steps, len(FED_DISTRICTS))

    global_weights: list[np.ndarray] | None = None
    local_weights: dict[str, list[np.ndarray]] = {}

    for round_index in range(rounds):
        updates: list[tuple[list[np.ndarray], int]] = []
        for junction in FED_DISTRICTS:
            weights, count = train_local(junction, global_weights, steps, seed)
            local_weights[junction] = weights
            updates.append((weights, count))
        # Flower's own FedAvg aggregation, not a hand-rolled mean.
        global_weights = aggregate(updates)
        log.info("  round %d/%d aggregated %d district updates",
                 round_index + 1, rounds, len(updates))

    results: dict[str, dict[str, float]] = {}
    log.info("  evaluating on held-out junction %s", FED_HELD_OUT)
    lanes = junction_lanes(FED_HELD_OUT)
    results["fixed"] = evaluate_fixed(FED_HELD_OUT, lanes, seed)
    results["fedavg"] = evaluate(FED_HELD_OUT, global_weights, seed)
    for junction in FED_DISTRICTS:
        results[f"local_{junction}"] = evaluate(FED_HELD_OUT, local_weights[junction], seed)

    # Identical metrics across structurally different policies is the exact
    # pattern Weeks 5 and 6 found, so it is measured rather than assumed: how far
    # apart are the weights, and how often do the policies actually disagree?
    divergence: dict[str, float] = {}
    for junction in FED_DISTRICTS:
        divergence[f"fedavg_vs_{junction}"] = weight_distance(
            global_weights, local_weights[junction])
    ordered = list(FED_DISTRICTS)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            divergence[f"{left}_vs_{right}"] = weight_distance(
                local_weights[left], local_weights[right])

    agreement: dict[str, dict[str, float]] = {}
    for junction in FED_DISTRICTS:
        agreement[f"fedavg_vs_{junction}"] = action_agreement(
            FED_HELD_OUT, global_weights, local_weights[junction], seed)

    log.info("  weight L2 distances: %s",
             ", ".join(f"{k}={v:.3f}" for k, v in divergence.items()))
    log.info("  action agreement on %s: %s", FED_HELD_OUT,
             ", ".join(f"{k}={v['agreement']:.3f}" for k, v in agreement.items()))

    local_keys = [f"local_{j}" for j in FED_DISTRICTS]
    results["local_mean"] = {
        key: float(np.mean([results[k][key] for k in local_keys]))
        for key in results["fedavg"]
    }

    for name, row in results.items():
        log.info("    %-12s junction wait %6.2fs  corridor wait %6.2fs",
                 name, row["junction_wait_s"], row["corridor_wait_s"])
    return {"seed": seed, "results": results,
            "weight_divergence": divergence, "action_agreement": agreement}


def plot(records: list[dict[str, Any]]) -> None:
    """Chart held-out performance for every configuration.

    Args:
        records: per-seed records.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["fixed"] + [f"local_{j}" for j in FED_DISTRICTS] + ["local_mean", "fedavg"]
    metrics = [("junction_wait_s", f"Wait at held-out junction {FED_HELD_OUT} (s)"),
               ("corridor_wait_s", "Corridor-wide wait (s)")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (key, label) in zip(axes, metrics):
        means, spreads = [], []
        for name in names:
            values = [r["results"][name][key] for r in records]
            means.append(float(np.mean(values)))
            spreads.append(float(np.std(values)))
        colours = ["#8a929e"] + ["#9fb8c4"] * len(FED_DISTRICTS) + ["#d18a12", "#1a7ba4"]
        ax.bar(range(len(names)), means, yerr=spreads, capsize=3,
               color=colours, alpha=0.93)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Week 9 federated learning - {len(records)}-seed means, "
                 f"trained on {', '.join(FED_DISTRICTS)}, evaluated on {FED_HELD_OUT}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FED_CHART, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", FED_CHART)


def main() -> None:
    """Run the federated experiment across seeds and report."""
    parser = argparse.ArgumentParser(description="Federated corridor signal control.")
    parser.add_argument("--rounds", type=int, default=FED_ROUNDS)
    parser.add_argument("--steps", type=int, default=FED_STEPS_PER_ROUND)
    parser.add_argument("--seeds", type=int, nargs="+", default=FED_SEEDS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(FED_RESULTS), exist_ok=True)
    records = [run_seed(seed, args.rounds, args.steps) for seed in args.seeds]
    plot(records)

    def mean_of(name: str, key: str) -> float:
        return float(np.mean([r["results"][name][key] for r in records]))

    fedavg = mean_of("fedavg", "junction_wait_s")
    local_mean = mean_of("local_mean", "junction_wait_s")
    fixed = mean_of("fixed", "junction_wait_s")
    improvement = (local_mean - fedavg) / local_mean * 100 if local_mean else 0.0

    log.info("")
    log.info("Held-out junction %s, %d-seed mean wait:", FED_HELD_OUT, len(records))
    log.info("  fixed program      %6.2f s", fixed)
    log.info("  local mean         %6.2f s", local_mean)
    log.info("  federated average  %6.2f s  (%+.1f%% vs local mean)",
             fedavg, -improvement)
    log.info("")
    log.info("DoD - federated averaging improves the held-out district: %s",
             "MET" if fedavg < local_mean else "NOT MET")

    payload = {
        "districts": FED_DISTRICTS,
        "held_out": FED_HELD_OUT,
        "rounds": args.rounds,
        "steps_per_round": args.steps,
        "seeds": args.seeds,
        "per_seed": records,
        "summary": {
            "fixed_wait_s": fixed,
            "local_mean_wait_s": local_mean,
            "fedavg_wait_s": fedavg,
            "improvement_pct_vs_local_mean": improvement,
            "dod_met": bool(fedavg < local_mean),
        },
    }
    with open(FED_RESULTS, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("Wrote %s", FED_RESULTS)


if __name__ == "__main__":
    main()
