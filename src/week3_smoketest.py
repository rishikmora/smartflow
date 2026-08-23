"""Smoke test the Week 3 single-agent corridor environment."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any

import numpy as np
import sumolib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_config import CORRIDOR_NET, CORRIDOR_ROUTE, CORRIDOR_TLS_ID
from week3_config import DELTA_TIME, MAX_GREEN, MIN_GREEN, SIM_SECONDS, YELLOW_TIME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def list_signalized_junctions() -> list[tuple[str, int]]:
    """Return traffic-light IDs and connected edge counts from the corridor net."""
    try:
        net = sumolib.net.readNet(CORRIDOR_NET)
    except Exception as exc:
        raise RuntimeError(f"Could not read SUMO network {CORRIDOR_NET}: {exc}") from exc
    return [
        (node.getID(), len(node.getIncoming()) + len(node.getOutgoing()))
        for node in net.getNodes()
        if node.getType() == "traffic_light"
    ]


def _build_env() -> Any:
    """Create the single-agent sumo-rl environment for the selected junction."""
    if not os.environ.get("SUMO_HOME"):
        raise EnvironmentError("SUMO_HOME is not set; set it to your SUMO install before running.")
    try:
        from sumo_rl import SumoEnvironment
    except Exception as exc:
        raise RuntimeError(f"Could not import sumo_rl after SUMO_HOME check: {exc}") from exc
    return SumoEnvironment(
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
        fixed_ts=False,
        sumo_seed=0,
        reward_fn="diff-waiting-time",
        add_system_info=True,
    )


def run_smoke_test(steps: int = 20) -> None:
    """Reset the corridor env and execute random actions for a few steps."""
    log.info("Signalized junction inventory: %s", list_signalized_junctions())
    log.info("Selected CORRIDOR_TLS_ID=%s", CORRIDOR_TLS_ID)
    env = _build_env()
    try:
        obs, _ = env.reset(seed=0)
        log.info("Initial observation shape: %s", np.asarray(obs).shape)
        for idx in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            reward_value = float(reward)
            if math.isnan(reward_value):
                raise ValueError(f"NaN reward at step {idx}")
            log.info(
                "step=%02d action=%s obs_shape=%s reward=%.4f",
                idx + 1,
                action,
                np.asarray(obs).shape,
                reward_value,
            )
            if terminated or truncated:
                log.info("Episode ended during smoke test: terminated=%s truncated=%s", terminated, truncated)
                break
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    run_smoke_test()


if __name__ == "__main__":
    main()
