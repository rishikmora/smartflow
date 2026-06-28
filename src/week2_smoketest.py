"""
Week 2 Day 1 smoketest.

Instantiates the sumo-rl single-intersection benchmark environment,
resets it, takes 50 random actions, and prints the observation shape
and reward at each step.

DoD: 50 steps complete, obs shape printed, no NaN rewards, no crash.
"""
import logging
import sys

from sumo_rl import SumoEnvironment

from week2_config import (
    BENCHMARK_NET,
    BENCHMARK_ROUTE,
    DELTA_TIME,
    MIN_GREEN,
    MAX_GREEN,
    SIM_SECONDS,
    YELLOW_TIME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    """Run 50 random-action steps and report obs shape + rewards."""
    log.info("Creating SumoEnvironment (single-agent, no GUI)...")
    env = SumoEnvironment(
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
    )

    obs, info = env.reset(seed=0)
    log.info("obs shape : %s", obs.shape)
    log.info("obs dtype : %s", obs.dtype)
    log.info("action space: %s", env.action_space)

    print(f"\n{'Step':>4}  {'Reward':>10}  {'Terminated':>10}  {'Truncated':>9}")
    print("-" * 42)

    nan_found = False
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != reward:        # NaN check
            nan_found = True
        print(f"{step:>4}  {reward:>10.4f}  {str(terminated):>10}  {str(truncated):>9}")

        if terminated or truncated:
            log.info("Episode ended at step %d — resetting.", step)
            obs, info = env.reset(seed=0)

    env.close()

    if nan_found:
        log.error("NaN reward detected — environment wiring is broken.")
        sys.exit(1)

    log.info("50 steps complete. Obs shape=%s. No NaN rewards.", obs.shape)
    print("\nSUCCESS")


if __name__ == "__main__":
    main()
