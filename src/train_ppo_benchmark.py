"""
PPO training script for the sumo-rl 2-way single-intersection benchmark.

Trains in chunks of PPO_CHUNK_SIZE steps, closing and re-creating the SUMO
process between chunks.  This avoids the "Socket reset by peer" crash that
occurs when SUMO accumulates memory over 60+ consecutive episodes on Windows.
Intermediate checkpoints are saved every 10k steps so a partial run can be
resumed without restarting from scratch.

Usage:
    # Day 2 - short pipeline validation (~5 min)
    python src/train_ppo_benchmark.py --timesteps 30000 --seed 0 --tag short

    # Day 4 - full 3-seed training
    python src/train_ppo_benchmark.py --seed 0
    python src/train_ppo_benchmark.py --seed 1
    python src/train_ppo_benchmark.py --seed 2

Saves:
    models/ppo_benchmark_seed{N}.zip          (final model)
    outputs/checkpoints/ppo_seed{N}/          (intermediate checkpoints)
    outputs/ppo_seed{N}_hparams.json          (hyperparameters)
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from sumo_rl import SumoEnvironment

from week2_config import (
    BENCHMARK_NET,
    BENCHMARK_ROUTE,
    DELTA_TIME,
    MAX_GREEN,
    MIN_GREEN,
    MODELS_DIR,
    OUTPUTS_DIR,
    PPO_CHUNK_SIZE,
    PPO_TIMESTEPS_FULL,
    PPO_TIMESTEPS_SHORT,
    SIM_SECONDS,
    YELLOW_TIME,
    model_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHECKPOINT_BASE = os.path.join(OUTPUTS_DIR, "checkpoints")


class ProgressCallback(BaseCallback):
    """Logs training progress every N steps with ETA."""

    def __init__(self, total_timesteps: int, log_interval: int = 5000) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._last_log = 0
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            elapsed = time.perf_counter() - self._t0
            rate = self.num_timesteps / max(elapsed, 1e-6)
            remaining = (self.total_timesteps - self.num_timesteps) / max(rate, 1e-6)
            log.info(
                "step %6d / %d  |  %.0f steps/s  |  ETA %.0f s",
                self.num_timesteps,
                self.total_timesteps,
                rate,
                remaining,
            )
            self._last_log = self.num_timesteps
        return True


def build_env(seed: int) -> Monitor:
    """Create and wrap the benchmark environment for a given seed.

    Args:
        seed: SUMO random seed for this training chunk.

    Returns:
        Gymnasium-compatible Monitor-wrapped SumoEnvironment.
    """
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
        sumo_seed=seed,
        reward_fn="diff-waiting-time",
    )
    log_dir = os.path.join(OUTPUTS_DIR, "monitor")
    os.makedirs(log_dir, exist_ok=True)
    return Monitor(env, filename=os.path.join(log_dir, f"seed{seed}"))


def _checkpoint_dir(seed: int) -> str:
    """Return the directory where intermediate checkpoints are stored."""
    return os.path.join(CHECKPOINT_BASE, f"ppo_seed{seed}")


def _find_last_checkpoint(seed: int) -> tuple[int, str | None]:
    """Scan the checkpoint directory for the latest saved model.

    Args:
        seed: training seed to look up.

    Returns:
        Tuple of (steps_completed, checkpoint_path).  steps_completed is 0
        and checkpoint_path is None if no checkpoint exists.
    """
    ckpt_dir = _checkpoint_dir(seed)
    pattern = os.path.join(ckpt_dir, f"ppo_seed{seed}_*.zip")
    files = glob.glob(pattern)
    if not files:
        return 0, None

    best_steps = 0
    best_path = None
    for f in files:
        m = re.search(r"_(\d+)_steps\.zip$", f)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = f
    return best_steps, best_path


def train(seed: int, timesteps: int, tag: str = "") -> str:
    """Train a PPO agent in fixed-size chunks to avoid SUMO memory crashes.

    Each chunk runs for PPO_CHUNK_SIZE steps, after which the SUMO process is
    closed and a fresh environment is created for the next chunk.  SB3's
    ``reset_num_timesteps=False`` keeps the global step counter and replay
    buffer state continuous across chunk boundaries.

    If a checkpoint from a previous (interrupted) run exists, training resumes
    from that checkpoint rather than starting over.

    Args:
        seed: random seed for both SUMO and SB3.
        timesteps: total training budget.
        tag: optional label appended to file names (e.g. "short" for Day 2).

    Returns:
        Path to the saved final model .zip file.
    """
    label = f"seed{seed}" + (f"_{tag}" if tag else "")
    final_path = (
        model_path(seed)
        if not tag
        else os.path.join(MODELS_DIR, f"ppo_benchmark_{label}.zip")
    )
    ckpt_dir = _checkpoint_dir(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    hparams = {
        "algorithm": "PPO",
        "seed": seed,
        "timesteps": timesteps,
        "chunk_size": PPO_CHUNK_SIZE,
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "reward_fn": "diff-waiting-time",
        "delta_time": DELTA_TIME,
        "sim_seconds": SIM_SECONDS,
        "benchmark_net": BENCHMARK_NET,
        "benchmark_route": BENCHMARK_ROUTE,
    }

    # ── resume from last checkpoint if one exists ────────────────────────────
    steps_done, last_ckpt = _find_last_checkpoint(seed)
    if last_ckpt:
        log.info("Resuming from checkpoint: %s  (steps_done=%d)", last_ckpt, steps_done)
    else:
        log.info("No checkpoint found — starting from scratch.")
    steps_remaining = max(0, timesteps - steps_done)
    if steps_remaining == 0:
        log.info("Training already complete for seed=%d. Loading final model.", seed)
        return final_path

    log.info(
        "Training PPO  seed=%d  total=%d  remaining=%d  chunk=%d",
        seed, timesteps, steps_remaining, PPO_CHUNK_SIZE,
    )
    t0_total = time.perf_counter()
    model: PPO | None = None

    while steps_done < timesteps:
        chunk = min(PPO_CHUNK_SIZE, timesteps - steps_done)
        log.info(
            "--- Chunk start: steps %d -> %d ---", steps_done, steps_done + chunk
        )

        env = build_env(seed)

        if model is None:
            if last_ckpt:
                log.info("Loading checkpoint weights from %s", last_ckpt)
                model = PPO.load(last_ckpt, env=env)
            else:
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    seed=seed,
                    learning_rate=hparams["learning_rate"],
                    n_steps=hparams["n_steps"],
                    batch_size=hparams["batch_size"],
                    n_epochs=hparams["n_epochs"],
                    gamma=hparams["gamma"],
                    gae_lambda=hparams["gae_lambda"],
                    clip_range=hparams["clip_range"],
                    ent_coef=hparams["ent_coef"],
                    verbose=0,
                )
        else:
            model.set_env(env)

        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=ckpt_dir,
            name_prefix=f"ppo_seed{seed}",
        )
        progress_cb = ProgressCallback(timesteps)

        t0_chunk = time.perf_counter()
        model.learn(
            total_timesteps=chunk,
            callback=[progress_cb, checkpoint_cb],
            reset_num_timesteps=(steps_done == 0 and last_ckpt is None),
        )
        elapsed_chunk = time.perf_counter() - t0_chunk

        steps_done += chunk
        log.info(
            "Chunk done in %.0f s  |  steps_done=%d / %d",
            elapsed_chunk, steps_done, timesteps,
        )
        env.close()

    elapsed_total = time.perf_counter() - t0_total
    log.info(
        "Training complete in %.0f s (%.0f steps/s)",
        elapsed_total, timesteps / elapsed_total,
    )

    model.save(final_path)
    log.info("Model saved -> %s", final_path)

    hparams_path = final_path.replace(".zip", "_hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    log.info("Hyperparameters saved -> %s", hparams_path)

    return final_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train PPO on the benchmark intersection (chunked, crash-safe)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timesteps", type=int, default=PPO_TIMESTEPS_FULL,
        help="Total training budget (default from week2_config.py).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional suffix for file names (e.g. 'short' for Day 2).",
    )
    args = parser.parse_args()
    train(seed=args.seed, timesteps=args.timesteps, tag=args.tag)


if __name__ == "__main__":
    main()
