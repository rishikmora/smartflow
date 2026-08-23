"""Week 3 configuration for single-agent PPO on the corridor network."""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CONFIGS_DIR = os.path.join(ROOT, "configs")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")

CORRIDOR_NET = os.path.join(DATA_DIR, "corridor.net.xml")
CORRIDOR_ACTUATED_NET = os.path.join(DATA_DIR, "corridor_actuated.net.xml")
CORRIDOR_ROUTE = os.path.join(DATA_DIR, "corridor.rou.xml")
CORRIDOR_FIXED_CONFIG = os.path.join(CONFIGS_DIR, "corridor.sumocfg")
CORRIDOR_ACTUATED_CONFIG = os.path.join(CONFIGS_DIR, "corridor_actuated.sumocfg")

# Chosen from the traffic-light inventory by maximum connected lane count.
# B1, B2, C1, and C2 tie; B1 is used as the stable representative.
CORRIDOR_TLS_ID = "B1"
CORRIDOR_TLS_IDS = ["A1", "B0", "A2", "B1", "B2", "B3", "C0", "C1", "C2", "C3", "D1", "D2"]

SIM_SECONDS = 1800
DELTA_TIME = 5
YELLOW_TIME = 2
MIN_GREEN = 5
MAX_GREEN = 90

PPO_TIMESTEPS_SHORT = 30_000
PPO_TIMESTEPS_FULL = 200_000
PPO_CHUNK_SIZE = 20_000
TRAIN_SEEDS = [0, 1, 2]

WEEK3_CSV = os.path.join(OUTPUTS_DIR, "week3_corridor_metrics.csv")
WEEK3_SHORT_RUN_PNG = os.path.join(OUTPUTS_DIR, "week3_short_run.png")
WEEK3_PNG = os.path.join(OUTPUTS_DIR, "week3_comparison.png")
WEEK3_REPORT = os.path.join(OUTPUTS_DIR, "week3_report.md")
WEEK3_DEFERRED = os.path.join(OUTPUTS_DIR, "week3_deferred.md")

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


def model_path(seed: int) -> str:
    """Return the Week 3 PPO checkpoint path for a seed."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"ppo_corridor_seed{seed}.zip")
