"""Smoke test sumo-rl's PettingZoo parallel multi-agent corridor API."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week4_config import CORRIDOR_NET, CORRIDOR_ROUTE, DELTA_TIME, MAX_GREEN, MIN_GREEN, SIM_SECONDS, YELLOW_TIME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def _build_env() -> Any:
    """Create a parallel PettingZoo environment for all traffic lights."""
    if not os.environ.get("SUMO_HOME"):
        raise EnvironmentError("SUMO_HOME is not set; set it before running SUMO smoke tests.")
    try:
        from sumo_rl import parallel_env
    except Exception as exc:
        raise RuntimeError(f"sumo_rl.parallel_env is unavailable: {exc}") from exc
    return parallel_env(
        net_file=CORRIDOR_NET,
        route_file=CORRIDOR_ROUTE,
        out_csv_name=None,
        num_seconds=SIM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        use_gui=False,
        sumo_warnings=False,
        reward_fn="diff-waiting-time",
    )


def run_smoke_test(steps: int = 20) -> None:
    """Reset and step all agents with random actions."""
    env = _build_env()
    try:
        observations, _ = env.reset(seed=0)
        log.info("agents=%s", env.agents)
        for agent, obs in observations.items():
            log.info("agent=%s obs_shape=%s", agent, np.asarray(obs).shape)
        for idx in range(steps):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(actions)
            for agent, reward in rewards.items():
                if math.isnan(float(reward)):
                    raise ValueError(f"NaN reward for {agent} at step {idx}")
            log.info("step=%02d rewards=%s", idx + 1, {k: round(float(v), 3) for k, v in rewards.items()})
            if all(terminations.values()) or all(truncations.values()):
                break
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    run_smoke_test()


if __name__ == "__main__":
    main()
