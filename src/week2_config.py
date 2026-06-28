"""
Week 2 central config — all scripts import paths and constants from here.
Do NOT scatter net/route paths or hyperparameters across individual scripts.
"""
import os
import sumo_rl

# ── benchmark network (sumo-rl bundled 2-way single intersection) ─────────────
_NETS_DIR = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2way-single-intersection")

BENCHMARK_NET   = os.path.join(_NETS_DIR, "single-intersection.net.xml")
BENCHMARK_ROUTE = os.path.join(_NETS_DIR, "single-intersection-vhvh.rou.xml")

# ── simulation settings ───────────────────────────────────────────────────────
SIM_SECONDS  = 3600   # episode length (seconds)
DELTA_TIME   = 5      # seconds between agent decisions
YELLOW_TIME  = 2      # yellow-phase duration (must be < DELTA_TIME)
MIN_GREEN    = 5      # minimum green time per phase
MAX_GREEN    = 50     # maximum green time per phase

# ── training settings ─────────────────────────────────────────────────────────
PPO_TIMESTEPS_SHORT = 30_000    # Day 2 pipeline-validation run
PPO_TIMESTEPS_FULL  = 200_000   # Day 4 real training run per seed
TRAIN_SEEDS = [0, 1, 2]

# ── output paths ─────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR  = os.path.join(ROOT, "models")
BENCHMARK_CSV = os.path.join(OUTPUTS_DIR, "week2_benchmark_metrics.csv")
BENCHMARK_PNG = os.path.join(OUTPUTS_DIR, "week2_benchmark_comparison.png")

CSV_HEADER = ["controller", "seed",
              "avg_wait_time_s", "max_queue_len", "throughput_veh"]


def model_path(seed: int) -> str:
    """Return the .zip path for a trained PPO model at the given seed."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"ppo_benchmark_seed{seed}.zip")
