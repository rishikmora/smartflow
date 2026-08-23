"""Week 3 configuration: single-agent PPO on one real corridor junction.

Week 2 proved the PPO pipeline on sumo-rl's standard 2-way single-intersection
benchmark. Week 3 ports that pipeline to one junction of the Week 1 corridor
(``data/corridor.net.xml``, a 12-junction grid) while the other 11 junctions keep
the fixed-time program defined in the network file.

Experimental design
-------------------
Four controllers are compared on the identical demand (``data/corridor.rou.xml``,
1800 vehicles over 1800 s):

===================  ============================================================
``fixed``            all 12 junctions fixed-time (Week 1 reference)
``actuated``        all 12 junctions SUMO-actuated (Week 1 reference)
``actuated_single``  only ``CORRIDOR_TLS_ID`` actuated, other 11 fixed-time
``ppo``              only ``CORRIDOR_TLS_ID`` under RL, other 11 fixed-time
===================  ============================================================

``actuated_single`` is the honest control for ``ppo``: it isolates "a smarter
controller at one junction" from "a smarter controller everywhere". Comparing a
single RL junction against a fully-actuated corridor would confound the two.

Metrics are logged at two scopes — ``junction`` (the controlled junction's incoming
lanes) and ``corridor`` (the whole network). The junction scope is the primary
Week 3 claim, because one junction out of twelve cannot be expected to move
corridor-wide numbers much; that limitation is what motivates Week 4's multi-agent
work.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
CONFIGS_DIR = os.path.join(ROOT, "configs")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# ── the junction placed under RL control ─────────────────────────────────────
# B1 is one of the four interior junctions (B1, B2, C1, C2) that carry four
# neighbours each and therefore the most conflicting demand. Interior junctions
# are where signal timing actually matters; edge junctions have less to gain.
CORRIDOR_TLS_ID = "B1"
CORRIDOR_TLS_IDS = ["A1", "A2", "B0", "B1", "B2", "B3", "C0", "C1", "C2", "C3", "D1", "D2"]

# ── network / demand ─────────────────────────────────────────────────────────
CORRIDOR_NET = os.path.join(DATA_DIR, "corridor.net.xml")
CORRIDOR_ACTUATED_NET = os.path.join(DATA_DIR, "corridor_actuated.net.xml")
CORRIDOR_SINGLE_ACTUATED_NET = os.path.join(DATA_DIR, f"corridor_actuated_{CORRIDOR_TLS_ID}.net.xml")
CORRIDOR_ROUTE = os.path.join(DATA_DIR, "corridor.rou.xml")

CORRIDOR_FIXED_CONFIG = os.path.join(CONFIGS_DIR, "corridor.sumocfg")
CORRIDOR_ACTUATED_CONFIG = os.path.join(CONFIGS_DIR, "corridor_actuated.sumocfg")
CORRIDOR_SINGLE_ACTUATED_CONFIG = os.path.join(CONFIGS_DIR, f"corridor_actuated_{CORRIDOR_TLS_ID}.sumocfg")

# ── simulation settings (shared by training and evaluation) ──────────────────
SIM_SECONDS = 1800   # matches the route file's demand window
DELTA_TIME = 5       # seconds between agent decisions
YELLOW_TIME = 2      # must be < DELTA_TIME
MIN_GREEN = 5
MAX_GREEN = 90

# ── training settings ────────────────────────────────────────────────────────
# Measured ~168 agent-steps/s for the single-junction corridor env on this machine,
# so 200k steps is roughly 20 min/seed and 556 episodes — well past the point where
# the Week 2 benchmark reward curve flattened.
PPO_TIMESTEPS_SHORT = 30_000
PPO_TIMESTEPS_FULL = 200_000
PPO_CHUNK_SIZE = 50_000   # restart SUMO between chunks to bound memory growth
TRAIN_SEEDS = [0, 1, 2]

PPO_HPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
}

# ── outputs ──────────────────────────────────────────────────────────────────
WEEK3_CSV = os.path.join(OUTPUTS_DIR, "week3_corridor_metrics.csv")
WEEK3_PNG = os.path.join(OUTPUTS_DIR, "week3_comparison.png")
WEEK3_CURVE_PNG = os.path.join(OUTPUTS_DIR, "week3_reward_curves.png")
WEEK3_REPORT = os.path.join(OUTPUTS_DIR, "week3_report.md")

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

BASELINE_NETS = {
    "fixed": CORRIDOR_NET,
    "actuated": CORRIDOR_ACTUATED_NET,
    "actuated_single": CORRIDOR_SINGLE_ACTUATED_NET,
}


def model_path(seed: int, tag: str = "") -> str:
    """Return the Week 3 PPO checkpoint path for a seed.

    Args:
        seed: training seed.
        tag: optional suffix, e.g. ``"short"`` for the pipeline-validation run.

    Returns:
        Absolute path to the ``.zip`` model file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return os.path.join(MODELS_DIR, f"ppo_corridor_seed{seed}{suffix}.zip")
