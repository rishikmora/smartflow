"""Train PPO on the Week 3 corridor single-agent environment."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_config import CORRIDOR_NET, CORRIDOR_ROUTE, DELTA_TIME, MAX_GREEN, MIN_GREEN
from week3_config import MODELS_DIR, OUTPUTS_DIR, PPO_CHUNK_SIZE, PPO_TIMESTEPS_FULL
from week3_config import SIM_SECONDS, WEEK3_SHORT_RUN_PNG, YELLOW_TIME, model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_BASE = os.path.join(OUTPUTS_DIR, "checkpoints")


def _require_training_imports() -> tuple[type, type, type, type]:
    """Import training dependencies after validating SUMO_HOME."""
    if not os.environ.get("SUMO_HOME"):
        raise EnvironmentError("SUMO_HOME is not set; set it before PPO training.")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from sumo_rl import SumoEnvironment
    except Exception as exc:
        raise RuntimeError(f"Could not import training dependencies: {exc}") from exc
    return PPO, BaseCallback, CheckpointCallback, Monitor, SumoEnvironment


class RewardCurveCallback:
    """Factory for a callback class that records rollout rewards."""

    @staticmethod
    def build(base_callback: type) -> type:
        """Create a BaseCallback subclass tied to the installed SB3 class."""

        class _RewardCurveCallback(base_callback):  # type: ignore[misc, valid-type]
            """Record mean reward snapshots during learning."""

            def __init__(self, total_timesteps: int, log_interval: int = 5000) -> None:
                super().__init__()
                self.total_timesteps = total_timesteps
                self.log_interval = log_interval
                self.points: list[tuple[int, float]] = []
                self._last_log = 0
                self._t0 = time.perf_counter()

            def _on_step(self) -> bool:
                if self.num_timesteps - self._last_log >= self.log_interval:
                    rewards = self.locals.get("rewards", [])
                    reward = float(rewards[0]) if len(rewards) else 0.0
                    self.points.append((self.num_timesteps, reward))
                    elapsed = time.perf_counter() - self._t0
                    rate = self.num_timesteps / max(elapsed, 1e-6)
                    log.info("step=%d/%d reward=%.4f rate=%.0f steps/s", self.num_timesteps, self.total_timesteps, reward, rate)
                    self._last_log = self.num_timesteps
                return True

        return _RewardCurveCallback


def _build_env(seed: int, monitor_cls: type, env_cls: type):
    """Create a Monitor-wrapped corridor env."""
    env = env_cls(
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
        sumo_seed=seed,
        reward_fn="diff-waiting-time",
        add_system_info=True,
    )
    log_dir = os.path.join(OUTPUTS_DIR, "monitor_week3")
    os.makedirs(log_dir, exist_ok=True)
    return monitor_cls(env, filename=os.path.join(log_dir, f"corridor_seed{seed}"))


def _checkpoint_dir(seed: int) -> str:
    """Return the checkpoint directory for one seed."""
    return os.path.join(CHECKPOINT_BASE, f"ppo_corridor_seed{seed}")


def _find_last_checkpoint(seed: int) -> tuple[int, str | None]:
    """Find the latest intermediate checkpoint for a seed."""
    pattern = os.path.join(_checkpoint_dir(seed), f"ppo_corridor_seed{seed}_*.zip")
    best_steps = 0
    best_path = None
    for path in glob.glob(pattern):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        if match and int(match.group(1)) > best_steps:
            best_steps = int(match.group(1))
            best_path = path
    return best_steps, best_path


def _save_reward_curve(points: list[tuple[int, float]], out_path: str) -> None:
    """Save a reward curve PNG from callback snapshots."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.8)
    ax.set_title("Week 3 PPO Short Run Reward Snapshots")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(seed: int, timesteps: int, tag: str = "") -> str:
    """Train PPO on the corridor and return the saved model path."""
    PPO, BaseCallback, CheckpointCallback, Monitor, SumoEnvironment = _require_training_imports()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(_checkpoint_dir(seed), exist_ok=True)

    final_path = model_path(seed) if not tag else os.path.join(MODELS_DIR, f"ppo_corridor_seed{seed}_{tag}.zip")
    steps_done, last_ckpt = _find_last_checkpoint(seed)
    steps_done = 0 if tag else steps_done
    last_ckpt = None if tag else last_ckpt
    reward_callback_cls = RewardCurveCallback.build(BaseCallback)
    reward_points: list[tuple[int, float]] = []
    model = None
    t0_total = time.perf_counter()

    while steps_done < timesteps:
        chunk = min(PPO_CHUNK_SIZE, timesteps - steps_done)
        env = _build_env(seed, Monitor, SumoEnvironment)
        try:
            if model is None:
                model = PPO.load(last_ckpt, env=env) if last_ckpt else PPO(
                    "MlpPolicy",
                    env,
                    seed=seed,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.0,
                    verbose=0,
                )
            else:
                model.set_env(env)
            reward_cb = reward_callback_cls(timesteps)
            checkpoint_cb = CheckpointCallback(
                save_freq=10_000,
                save_path=_checkpoint_dir(seed),
                name_prefix=f"ppo_corridor_seed{seed}",
            )
            model.learn(total_timesteps=chunk, callback=[reward_cb, checkpoint_cb], reset_num_timesteps=steps_done == 0)
            reward_points.extend(reward_cb.points)
            steps_done += chunk
        finally:
            env.close()

    model.save(final_path)
    hparams = {
        "algorithm": "PPO",
        "seed": seed,
        "timesteps": timesteps,
        "duration_s": round(time.perf_counter() - t0_total, 2),
        "net_file": CORRIDOR_NET,
        "route_file": CORRIDOR_ROUTE,
        "reward_fn": "diff-waiting-time",
    }
    with open(final_path.replace(".zip", "_hparams.json"), "w", encoding="utf-8") as handle:
        json.dump(hparams, handle, indent=2)
    if tag == "short":
        _save_reward_curve(reward_points, WEEK3_SHORT_RUN_PNG)
    return final_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Week 3 corridor PPO.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=PPO_TIMESTEPS_FULL)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    path = train(seed=args.seed, timesteps=args.timesteps, tag=args.tag)
    log.info("Saved model to %s", path)


if __name__ == "__main__":
    main()
