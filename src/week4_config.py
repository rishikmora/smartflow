"""Week 4 configuration for corridor multi-agent experiments."""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")

CORRIDOR_NET = os.path.join(DATA_DIR, "corridor.net.xml")
CORRIDOR_ROUTE = os.path.join(DATA_DIR, "corridor.rou.xml")
CORRIDOR_TLS_IDS = ["A1", "B0", "A2", "B1", "B2", "B3", "C0", "C1", "C2", "C3", "D1", "D2"]

SIM_SECONDS = 1800
DELTA_TIME = 5
YELLOW_TIME = 2
MIN_GREEN = 5
MAX_GREEN = 90

TRAIN_SEEDS = [0, 1, 2]
SHORT_TIMESTEPS = 50_000
FULL_TIMESTEPS = 500_000

WEEK4_METRICS_CSV = os.path.join(OUTPUTS_DIR, "week4_marl_metrics.csv")
WEEK4_NOTES = os.path.join(OUTPUTS_DIR, "week4_nonstationarity_notes.md")
WEEK4_REPORT = os.path.join(OUTPUTS_DIR, "week4_report.md")
