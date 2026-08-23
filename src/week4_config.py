"""Week 4 configuration: independent multi-agent PPO across the full corridor.

Week 3 put one junction under RL control and left eleven on fixed-time. Week 4
places **all twelve** junctions under RL control simultaneously, one independent
PPO policy per junction (RLlib multi-agent, ``policy_mapping_fn`` = identity).

That is the classic Independent Learners setting: from any single agent's point of
view the environment is non-stationary, because the other eleven policies keep
changing. Week 4's Definition of Done is deliberately modest — the corridor must
train end-to-end **without diverging** — and the non-stationarity notes record what
that instability actually looked like.

This module also carries the shared MARL constants (simulation timing, the graph
used by the GAT encoder) that Weeks 5 and 6 import.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

CORRIDOR_NET = os.path.join(DATA_DIR, "corridor.net.xml")
CORRIDOR_ROUTE = os.path.join(DATA_DIR, "corridor.rou.xml")

CORRIDOR_TLS_IDS = ["A1", "A2", "B0", "B1", "B2", "B3", "C0", "C1", "C2", "C3", "D1", "D2"]

# ── simulation settings (identical to Week 3 so results stay comparable) ─────
SIM_SECONDS = 1800
DELTA_TIME = 5
YELLOW_TIME = 2
MIN_GREEN = 5
MAX_GREEN = 90

# One episode is SIM_SECONDS / DELTA_TIME = 360 environment steps, and each
# environment step yields 12 agent steps (one per junction).
STEPS_PER_EPISODE = SIM_SECONDS // DELTA_TIME

# ── training budget ──────────────────────────────────────────────────────────
# Budgets are in *environment* steps, which is what RLlib counts. One env step is
# 12 agent steps (one decision per junction), so 48k env steps is 133 episodes and
# ~576k agent transitions.
#
# The figure comes from the Week 4 curves rather than from preference. Blocking those
# runs' median return into 9k-step chunks gives, for the three seeds:
#     seed 0: -648, -5.7, -3.8, -3.6, -3.3, -3.3, -3.4, -3.2
#     seed 1: -8.1, -442, -808, -4.4, -3.6, -3.2, -3.4, -6.0
#     seed 2: -5.9, -4.1, -3.7, -3.2, -2.8, -2.9, -3.1, -3.5
# Every seed has plateaued by ~27k steps (seed 1, the slowest, recovers there), so 48k
# leaves ~1.8x margin past the plateau.
#
# Week 4's own runs were executed at 72k, before this analysis existed. That is left
# as-is rather than re-run: it means the independent-learner baseline received *more*
# training than the shared policy compared against it in Week 5, which makes that
# comparison conservative rather than flattering.
SHORT_TIMESTEPS = 20_000
FULL_TIMESTEPS = 48_000
WEEK4_ACTUAL_TIMESTEPS = 72_000   # what the committed Week 4 checkpoints were trained at
TRAIN_SEEDS = [0, 1, 2]

# Each env runner owns one SUMO process. Keep (jobs x runners) at or below the core
# count — Ray sizes each cluster to the whole machine by default, and several jobs
# doing that at once spawns hundreds of workers that simply fight for cores.
NUM_ENV_RUNNERS = 5

PPO_CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.001,
    "num_epochs": 10,
    "minibatch_size": 512,
    "vf_clip_param": 10.0,
}

# ── GNN state embedding (PyTorch Geometric graph attention) ──────────────────
MAX_NEIGHBORS = 4          # the interior junctions (B1, B2, C1, C2) each feed four
GNN_HIDDEN_DIM = 64
GNN_EMBED_DIM = 32
GNN_HEADS = 4

# ── outputs ──────────────────────────────────────────────────────────────────
WEEK4_METRICS_CSV = os.path.join(OUTPUTS_DIR, "week4_marl_metrics.csv")
WEEK4_CURVE_PNG = os.path.join(OUTPUTS_DIR, "week4_training_curves.png")
WEEK4_PNG = os.path.join(OUTPUTS_DIR, "week4_comparison.png")
WEEK4_NOTES = os.path.join(OUTPUTS_DIR, "week4_nonstationarity_notes.md")
WEEK4_REPORT = os.path.join(OUTPUTS_DIR, "week4_report.md")

CSV_HEADER = [
    "controller",
    "seed",
    "scope",
    "tls_id",
    "avg_wait_time_s",
    "max_queue_len",
    "throughput_veh",
    "total_co2_kg",
]


def checkpoint_dir(mode: str, seed: int, tag: str = "") -> str:
    """Return the RLlib checkpoint directory for one training run.

    Args:
        mode: ``"independent"`` (Week 4) or ``"shared"`` (Week 5).
        seed: training seed.
        tag: optional run label.

    Returns:
        Absolute path; the directory is created if missing.
    """
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(MODELS_DIR, f"marl_{mode}_seed{seed}{suffix}")
    os.makedirs(path, exist_ok=True)
    return path


def training_log_path(mode: str, seed: int, tag: str = "") -> str:
    """Return the JSON path holding a run's per-iteration training history."""
    suffix = f"_{tag}" if tag else ""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    return os.path.join(OUTPUTS_DIR, f"week4_{mode}_seed{seed}{suffix}_training.json")
