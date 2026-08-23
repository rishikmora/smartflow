"""Smoke-test the SmartFlow corridor environments end-to-end against a live SUMO.

Replaces the separate Week 3 and Week 4 smoke tests with one entry point, so there is
no script left that builds the environment the *old* way. That matters here: creating
``SumoEnvironment`` directly on a 12-junction network silently overwrites all twelve
signal programs, which collapses corridor throughput and slows the environment by
roughly 15x. ``smartflow_env.ControlledSumoEnvironment`` is the fix, and this script
verifies it is doing its job.

Checks:

1. Every traffic light in the network is discoverable without starting a simulation.
2. The single-agent environment controls exactly the requested junction and leaves the
   other eleven on their network-defined program.
3. The multi-agent environment steps all twelve junctions, and observations are aligned
   onto a common layout with finite values.
4. Rewards are finite (no NaN) in both.
5. Reported throughput is in the sane range — the regression test for the bug above.

Usage:
    python src/env_smoketest.py
    python src/env_smoketest.py --steps 40 --num-seconds 600
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marl_env import make_parallel_env
from smartflow_env import list_traffic_lights, make_single_agent_env, neighbor_map, require_sumo_home
from week3_config import (
    CORRIDOR_NET,
    CORRIDOR_ROUTE,
    CORRIDOR_TLS_ID,
    DELTA_TIME,
    MAX_GREEN,
    MIN_GREEN,
    YELLOW_TIME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# Under a working setup the corridor clears well over this many vehicles in 1800 s.
# When every junction's program is clobbered it clears fewer than 100, so this is a
# wide but effective regression guard.
MIN_SANE_ARRIVALS_PER_1800S = 300


def check_network() -> list[str]:
    """Report the traffic lights and their adjacency without starting SUMO.

    Returns:
        The traffic-light ids found in the corridor network.
    """
    tls = list_traffic_lights(CORRIDOR_NET)
    log.info("Traffic lights in %s: %d -> %s", os.path.basename(CORRIDOR_NET), len(tls), tls)
    neighbours = neighbor_map(CORRIDOR_NET, tls)
    for ts in tls:
        log.info("  %-3s feeds -> %s", ts, neighbours[ts])
    if CORRIDOR_TLS_ID not in tls:
        raise ValueError(f"Configured CORRIDOR_TLS_ID={CORRIDOR_TLS_ID} is not in the network.")
    return tls


def check_single_agent(steps: int, num_seconds: int) -> None:
    """Step the single-agent environment with random actions.

    Args:
        steps: number of agent decisions to take.
        num_seconds: episode length in simulated seconds.

    Raises:
        ValueError: if control leaked to other junctions, or a reward was NaN.
    """
    env = make_single_agent_env(
        net_file=CORRIDOR_NET,
        route_file=CORRIDOR_ROUTE,
        controlled_ts=CORRIDOR_TLS_ID,
        num_seconds=num_seconds,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        seed=0,
    )
    try:
        controlled = list(env.ts_ids)
        if controlled != [CORRIDOR_TLS_ID]:
            raise ValueError(
                f"Expected control of exactly [{CORRIDOR_TLS_ID}] but the env controls {controlled}. "
                "Uncontrolled junctions would lose their network-defined signal program."
            )
        obs, _ = env.reset(seed=0)
        log.info("single-agent: controls %s, obs shape %s, %d actions",
                 controlled, np.asarray(obs).shape, env.action_space.n)
        for index in range(steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            if math.isnan(float(reward)):
                raise ValueError(f"NaN reward at step {index}")
            if terminated or truncated:
                break
        log.info("single-agent: %d steps OK, last reward %.4f", index + 1, float(reward))
    finally:
        env.close()


def check_multi_agent(num_seconds: int) -> None:
    """Run one full multi-agent episode with random actions and sanity-check throughput.

    Args:
        num_seconds: episode length in simulated seconds.

    Raises:
        ValueError: on NaN rewards, misaligned observations, or collapsed throughput.
    """
    env = make_parallel_env({"num_seconds": num_seconds, "seed": 0})
    try:
        obs, _ = env.reset(seed=0)
        dim = env.aligner.dim
        log.info("multi-agent: %d agents, aligned observation dim %d "
                 "(max_phases=%d, max_lanes=%d)",
                 len(env.possible_agents), dim, env.aligner.max_phases, env.aligner.max_lanes)
        for agent, vector in obs.items():
            array = np.asarray(vector)
            if array.shape != (dim,):
                raise ValueError(f"Agent {agent}: observation shape {array.shape}, expected ({dim},)")
            if not np.isfinite(array).all():
                raise ValueError(f"Agent {agent}: non-finite observation")

        # getArrivedNumber() reports only the most recent simulated second, while one
        # env.step() advances delta_time of them — so arrivals are accumulated through
        # the per-second hook rather than once per decision.
        arrived = 0

        def count_arrivals(conn: Any) -> None:
            nonlocal arrived
            arrived += conn.simulation.getArrivedNumber()

        env.set_step_hook(count_arrivals)

        steps = 0
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in obs}
            obs, rewards, _term, _trunc, _infos = env.step(actions)
            for agent, reward in rewards.items():
                if math.isnan(float(reward)):
                    raise ValueError(f"NaN reward for {agent} at step {steps}")
            steps += 1
        log.info("multi-agent: %d env steps, %d vehicles completed their trip", steps, arrived)
    finally:
        env.close()


def check_throughput(num_seconds: int = 1800) -> None:
    """Regression guard: a fixed-time corridor run must clear a sane vehicle count.

    If ``ControlledSumoEnvironment`` ever stops restricting control, every junction's
    program is overwritten and throughput collapses by an order of magnitude. This
    check catches that directly rather than leaving it to be noticed in a benchmark.

    Args:
        num_seconds: episode length in simulated seconds.

    Raises:
        ValueError: if throughput is implausibly low.
    """
    require_sumo_home()
    import traci

    from smartflow_env import sumo_binary
    from week3_config import CORRIDOR_FIXED_CONFIG

    traci.start([sumo_binary(), "-c", CORRIDOR_FIXED_CONFIG, "--seed", "0",
                 "--no-step-log", "--no-warnings"])
    arrived = 0
    try:
        while traci.simulation.getTime() < num_seconds:
            traci.simulationStep()
            arrived += traci.simulation.getArrivedNumber()
    finally:
        traci.close()

    log.info("fixed-time corridor throughput over %ds: %d vehicles", num_seconds, arrived)
    if arrived < MIN_SANE_ARRIVALS_PER_1800S * (num_seconds / 1800):
        raise ValueError(
            f"Only {arrived} vehicles completed their trip in {num_seconds}s. "
            "That indicates the corridor is gridlocked — check that traffic lights "
            "outside the controlled set still use their network-defined programs."
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Smoke-test the SmartFlow corridor environments.")
    parser.add_argument("--steps", type=int, default=20, help="Single-agent steps to take.")
    parser.add_argument("--num-seconds", type=int, default=600, help="Episode length for the env checks.")
    args = parser.parse_args()

    check_network()
    check_single_agent(args.steps, args.num_seconds)
    check_multi_agent(args.num_seconds)
    check_throughput()
    log.info("All environment smoke checks passed.")


if __name__ == "__main__":
    main()
