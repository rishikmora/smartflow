"""Week 3: train single-agent PPO on one junction of the SmartFlow corridor.

Ports the Week 2 pipeline (Stable-Baselines3 PPO, sumo-rl's default observation and
``diff-waiting-time`` reward) from the standard benchmark intersection onto junction
``week3_config.CORRIDOR_TLS_ID`` of ``data/corridor.net.xml``. The other eleven
junctions keep the fixed-time program defined in the network file — see
``smartflow_env.ControlledSumoEnvironment`` for why that needs an explicit subclass.

Training runs in chunks, closing and restarting the SUMO process between them, which
bounds SUMO's memory growth over the several hundred episodes a full run takes.
Intermediate checkpoints let an interrupted run resume instead of restarting.

Usage:
    # pipeline validation (~3 min)
    python src/train_ppo_corridor.py --seed 0 --timesteps 30000 --tag short

    # full run, one seed
    python src/train_ppo_corridor.py --seed 0

Produces:
    models/ppo_corridor_seed{N}.zip            trained policy
    models/ppo_corridor_seed{N}_hparams.json   seed + hyperparameters + wall time
    outputs/monitor_week3/corridor_seed{N}.monitor.csv   per-episode returns
    outputs/checkpoints/ppo_corridor_seed{N}/  intermediate checkpoints
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartflow_env import make_single_agent_env, require_sumo_home
from week3_config import (
    CORRIDOR_NET,
    CORRIDOR_ROUTE,
    CORRIDOR_TLS_ID,
    DELTA_TIME,
    MAX_GREEN,
    MIN_GREEN,
    MODELS_DIR,
    OUTPUTS_DIR,
    PPO_CHUNK_SIZE,
    PPO_HPARAMS,
    PPO_TIMESTEPS_FULL,
    SIM_SECONDS,
    YELLOW_TIME,
    model_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_BASE = os.path.join(OUTPUTS_DIR, "checkpoints")
MONITOR_DIR = os.path.join(OUTPUTS_DIR, "monitor_week3")


def _import_sb3() -> tuple[Any, Any, Any, Any]:
    """Import Stable-Baselines3 pieces after checking SUMO_HOME.

    Returns:
        ``(PPO, BaseCallback, CheckpointCallback, Monitor)``.

    Raises:
        RuntimeError: if Stable-Baselines3 cannot be imported.
    """
    require_sumo_home()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not import stable_baselines3: {exc}. "
            "Activate the venv and check 'pip install stable_baselines3[extra]'."
        ) from exc
    return PPO, BaseCallback, CheckpointCallback, Monitor


def _make_progress_callback(base_callback: type) -> type:
    """Build a progress-logging callback bound to the installed SB3 ``BaseCallback``.

    SB3's ``BaseCallback`` must be imported lazily, so the subclass is created at
    call time rather than at module import.

    Args:
        base_callback: the ``stable_baselines3.common.callbacks.BaseCallback`` class.

    Returns:
        A callback class that logs throughput and ETA.
    """

    class _ProgressCallback(base_callback):  # type: ignore[misc, valid-type]
        """Log training rate and ETA at a fixed step interval."""

        def __init__(self, total_timesteps: int, chunk_start: int = 0, log_interval: int = 10_000) -> None:
            """Track progress across chunked training runs.

            Args:
                total_timesteps: the full budget across all chunks.
                chunk_start: steps completed before this chunk, used only to measure
                    this chunk's throughput.
                log_interval: how often (in steps) to emit a progress line.
            """
            super().__init__()
            self.total_timesteps = total_timesteps
            self.chunk_start = chunk_start
            self.log_interval = log_interval
            self._last_log = 0
            self._t0 = time.perf_counter()

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_log < self.log_interval:
                return True
            # SB3's num_timesteps is already cumulative across chunks when
            # reset_num_timesteps=False (and after PPO.load), so it must not have the
            # chunk offset added to it again.
            done = self.num_timesteps
            elapsed = time.perf_counter() - self._t0
            rate = max(done - self.chunk_start, 0) / max(elapsed, 1e-6)
            remaining = max(self.total_timesteps - done, 0) / max(rate, 1e-6)
            log.info(
                "step %6d / %d | %.0f steps/s | ETA %.0f s",
                done, self.total_timesteps, rate, remaining,
            )
            self._last_log = self.num_timesteps
            return True

    return _ProgressCallback


def _checkpoint_dir(seed: int, tag: str = "") -> str:
    """Return the intermediate-checkpoint directory for one seed."""
    suffix = f"_{tag}" if tag else ""
    return os.path.join(CHECKPOINT_BASE, f"ppo_corridor_seed{seed}{suffix}")


def _find_last_checkpoint(seed: int, tag: str = "") -> tuple[int, str | None]:
    """Find the furthest-along checkpoint for a seed, so a run can resume.

    Args:
        seed: training seed.
        tag: optional run tag.

    Returns:
        ``(steps_completed, checkpoint_path)``; ``(0, None)`` when none exists.
    """
    prefix = f"ppo_corridor_seed{seed}{'_' + tag if tag else ''}"
    pattern = os.path.join(_checkpoint_dir(seed, tag), f"{prefix}_*.zip")
    best_steps, best_path = 0, None
    for path in glob.glob(pattern):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        if match and int(match.group(1)) > best_steps:
            best_steps, best_path = int(match.group(1)), path
    return best_steps, best_path


def build_env(seed: int, monitor_cls: type, tag: str = "") -> Any:
    """Create the Monitor-wrapped corridor training environment.

    Args:
        seed: SUMO seed for this chunk.
        monitor_cls: SB3's ``Monitor`` class.
        tag: optional run tag, used in the monitor filename.

    Returns:
        A ``Monitor``-wrapped single-agent environment controlling ``CORRIDOR_TLS_ID``.
    """
    env = make_single_agent_env(
        net_file=CORRIDOR_NET,
        route_file=CORRIDOR_ROUTE,
        controlled_ts=CORRIDOR_TLS_ID,
        num_seconds=SIM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        seed=seed,
        reward_fn="diff-waiting-time",
        add_system_info=False,   # network-wide polling is eval-only; too slow to train with
    )
    os.makedirs(MONITOR_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return monitor_cls(env, filename=os.path.join(MONITOR_DIR, f"corridor_seed{seed}{suffix}"))


def train(seed: int, timesteps: int, tag: str = "") -> str:
    """Train PPO on the corridor junction and save the final policy.

    Args:
        seed: seed for SUMO and for SB3's RNG.
        timesteps: total training budget in agent steps.
        tag: optional label appended to output filenames.

    Returns:
        Path to the saved ``.zip`` model.
    """
    PPO, BaseCallback, CheckpointCallback, Monitor = _import_sb3()
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_dir = _checkpoint_dir(seed, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    final_path = model_path(seed, tag)
    progress_cls = _make_progress_callback(BaseCallback)

    steps_done, last_ckpt = _find_last_checkpoint(seed, tag)
    if last_ckpt:
        log.info("Resuming seed=%d from %s (steps_done=%d)", seed, last_ckpt, steps_done)
    if steps_done >= timesteps:
        log.info("Seed %d already trained to %d steps; nothing to do.", seed, steps_done)
        return final_path

    log.info(
        "Training PPO seed=%d junction=%s budget=%d (remaining=%d, chunk=%d)",
        seed, CORRIDOR_TLS_ID, timesteps, timesteps - steps_done, PPO_CHUNK_SIZE,
    )
    model = None
    t0_total = time.perf_counter()

    while steps_done < timesteps:
        chunk = min(PPO_CHUNK_SIZE, timesteps - steps_done)
        env = build_env(seed, Monitor, tag)
        try:
            if model is None:
                if last_ckpt:
                    model = PPO.load(last_ckpt, env=env)
                else:
                    model = PPO(env=env, seed=seed, verbose=0, **PPO_HPARAMS)
            else:
                model.set_env(env)

            callbacks = [
                progress_cls(timesteps, chunk_start=steps_done),
                CheckpointCallback(
                    save_freq=25_000,
                    save_path=ckpt_dir,
                    name_prefix=f"ppo_corridor_seed{seed}{'_' + tag if tag else ''}",
                ),
            ]
            model.learn(
                total_timesteps=chunk,
                callback=callbacks,
                reset_num_timesteps=(steps_done == 0 and last_ckpt is None),
            )
            steps_done += chunk
            log.info("Chunk complete: %d / %d steps", steps_done, timesteps)
        finally:
            env.close()

    duration = time.perf_counter() - t0_total
    model.save(final_path)

    hparams = {
        "algorithm": "PPO",
        "seed": seed,
        "timesteps": timesteps,
        "tag": tag or None,
        "controlled_tls": CORRIDOR_TLS_ID,
        "uncontrolled_tls_behaviour": "native fixed-time program from corridor.net.xml",
        "reward_fn": "diff-waiting-time (sumo-rl default)",
        "observation_fn": "DefaultObservationFunction (sumo-rl default)",
        "delta_time": DELTA_TIME,
        "yellow_time": YELLOW_TIME,
        "min_green": MIN_GREEN,
        "max_green": MAX_GREEN,
        "sim_seconds": SIM_SECONDS,
        "net_file": CORRIDOR_NET,
        "route_file": CORRIDOR_ROUTE,
        "duration_s": round(duration, 1),
        "steps_per_s": round(timesteps / max(duration, 1e-6), 1),
        **PPO_HPARAMS,
    }
    with open(final_path.replace(".zip", "_hparams.json"), "w", encoding="utf-8") as handle:
        json.dump(hparams, handle, indent=2)

    log.info("Saved %s (%.0f s, %.0f steps/s)", final_path, duration, timesteps / max(duration, 1e-6))
    return final_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Week 3 corridor PPO on one junction.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=PPO_TIMESTEPS_FULL)
    parser.add_argument("--tag", type=str, default="", help="Optional run label, e.g. 'short'.")
    args = parser.parse_args()
    train(seed=args.seed, timesteps=args.timesteps, tag=args.tag)


if __name__ == "__main__":
    main()
