"""Week 8 configuration: vision, anomaly detection and the scenario planner.

Every Week 8 script imports its paths and constants from here rather than
re-deriving them, the same convention Weeks 2-7 follow.

Three components, three independent Definitions of Done:

1. **Vision** - a YOLOv8 detector trained on frames rendered by sumo-gui and
   labelled automatically from TraCI ground truth. The roadmap's original plan
   was UA-DETRAC plus rendered frames; UA-DETRAC needs a ~5 GB registered
   download that this machine cannot perform unattended, so the rendered half
   of that substitution carries the week. See ``outputs/week8_report.md`` for
   what that does and does not demonstrate.
2. **Anomaly detection** - a z-score detector run over the eval harness's own
   per-second metric stream, scored against injected anomalies with known
   ground truth.
3. **Scenario planner** - lane closures and weather perturbations applied to
   the committed corridor, run end to end through the existing harness.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")

# ── vision ──────────────────────────────────────────────────────────────────
VISION_DIR = os.path.join(OUTPUTS_DIR, "vision")
FRAMES_DIR = os.path.join(VISION_DIR, "dataset")
VISION_MODEL_DIR = os.path.join(MODELS_DIR, "yolo")
VISION_REPORT = os.path.join(OUTPUTS_DIR, "week8_vision_report.md")

SUMO_GUI_BIN = os.path.join(os.environ.get("SUMO_HOME", ""), "bin", "sumo-gui.exe")

# The GUI canvas is smaller than the requested window because of the toolbar, and
# SUMO widens the requested boundary to match the canvas aspect ratio. Both are
# handled by reading gui.getBoundary() back rather than assuming the request took.
GUI_WINDOW = (1000, 1000)

# Half-width, in SUMO metres, of the square view centred on a junction. 90 m puts
# a junction and its four approaches in frame with vehicles big enough to detect.
VIEW_HALF_M = 90.0

# Junctions the detector trains on, and the ones it is tested on. Holding out
# whole junctions rather than random frames is the honest split: a random split
# would leak, because consecutive frames of the same junction are near-duplicates.
TRAIN_JUNCTIONS = ["B1", "B2", "C1"]
HELD_OUT_JUNCTIONS = ["C2"]

# Frames per (junction, scenario) pair, and how many simulation seconds to skip
# between captures so consecutive frames are not near-identical.
FRAMES_PER_VIEW = 60
CAPTURE_EVERY_S = 12
WARMUP_S = 240

VISION_SCENARIOS = ["base", "peak"]

# Two classes so the detector has to classify as well as localise. Assignment is
# deterministic from the vehicle id, and applies only to these rendering runs -
# no benchmark result is produced from them.
VEHICLE_CLASSES = ["car", "truck"]
TRUCK_FRACTION = 0.18
TRUCK_LENGTH_M = 10.0
TRUCK_WIDTH_M = 2.5

YOLO_MODEL = "yolov8n.pt"
YOLO_EPOCHS = 40
YOLO_IMAGE_SIZE = 640
YOLO_BATCH = 8
YOLO_SEED = 0

# ── anomaly detection ───────────────────────────────────────────────────────
ANOMALY_REPORT = os.path.join(OUTPUTS_DIR, "week8_anomaly_report.md")
ANOMALY_CHART = os.path.join(OUTPUTS_DIR, "week8_anomalies.png")

# Rolling window, in samples, used to estimate the running mean and deviation.
ANOMALY_WINDOW = 60
ANOMALY_Z = 3.5
ANOMALY_MIN_SAMPLES = 20

# ── scenario planner ────────────────────────────────────────────────────────
PLANNER_REPORT = os.path.join(OUTPUTS_DIR, "week8_scenario_report.md")
PLANNER_CHART = os.path.join(OUTPUTS_DIR, "week8_scenarios.png")
PLANNER_CSV = os.path.join(OUTPUTS_DIR, "week8_scenario_metrics.csv")

# Weather is modelled the way SUMO itself recommends: not as graphics, but as the
# driving-behaviour changes rain actually causes - lower speeds, longer headways
# and reduced acceleration. Values are multipliers on the default vType.
WEATHER = {
    "clear": {"speed_factor": 1.00, "tau": 1.0, "decel": 1.00, "min_gap": 1.0},
    "rain": {"speed_factor": 0.80, "tau": 1.6, "decel": 0.75, "min_gap": 1.4},
    "fog": {"speed_factor": 0.65, "tau": 2.1, "decel": 0.65, "min_gap": 1.8},
}

# Closure scenarios: (label, edge id to restrict, description).
#
# Modelled as roadworks - the link is reduced to walking pace and made expensive
# to route over - rather than as a total ban. A total ban was tried first and
# rejected: many trips in corridor.rou.xml *end* on the closed edge, so banning
# it cancels them outright and the throughput collapse (1520 -> 520) measures
# impossible trips rather than the cost of rerouting. Roadworks degrade the link
# without making any trip infeasible, which is the question a planner is asking.
CLOSURE_SPEED = 2.0

CLOSURES = [
    ("none", None, "No restriction - reference run"),
    ("roadworks", "B1B2", "Central link B1B2 reduced to walking pace"),
]

PLANNER_SEEDS = [0, 1, 2]
PLANNER_SIM_SECONDS = 1800
