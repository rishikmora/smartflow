"""Week 9 configuration: federated learning, LoRA fine-tuning, priority routing.

Three components, three independent Definitions of Done:

1. **Federated learning** - each district trains a local signal policy on its own
   junction only, Flower's FedAvg averages them, and the averaged policy is
   evaluated on a district that took no part in training. The claim under test is
   that averaging beats the individual local policies on that held-out district.
2. **LoRA fine-tuning** - a small language model adapted on synthetic traffic
   Q&A, scored against the same model without the adapter.
3. **Priority routing and emission smoothing** - signal preemption for emergency
   vehicles, and an emissions term in the reward.

Districts are the four interior junctions. They are the only ones that share an
observation and action shape - ``obs=(11,) act=2`` - which federated averaging
requires, since averaging weights across different architectures is meaningless.
The eight perimeter junctions are three-way and observe ``(9,)``.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")

# ── federated learning ──────────────────────────────────────────────────────
FED_DIR = os.path.join(MODELS_DIR, "federated")
FED_CHART = os.path.join(OUTPUTS_DIR, "week9_federated.png")
FED_RESULTS = os.path.join(OUTPUTS_DIR, "week9_federated_results.json")

# Districts that train, and the district held out of training entirely.
FED_DISTRICTS = ["B1", "B2", "C1"]
FED_HELD_OUT = "C2"

FED_ROUNDS = 5
FED_STEPS_PER_ROUND = 6_000
FED_SEEDS = [0, 1, 2]
FED_EVAL_SECONDS = 1800

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 6,
    "gamma": 0.95,
    "verbose": 0,
}

# ── LoRA fine-tuning ────────────────────────────────────────────────────────
LORA_DIR = os.path.join(MODELS_DIR, "lora")
LORA_DATASET = os.path.join(OUTPUTS_DIR, "week9_qa_dataset.json")
LORA_RESULTS = os.path.join(OUTPUTS_DIR, "week9_lora_results.json")
LORA_CHART = os.path.join(OUTPUTS_DIR, "week9_lora.png")

# The roadmap names Phi-3-mini. This machine has no CUDA device and 3.8B
# parameters will not fine-tune on CPU in any reasonable time, so a much smaller
# model stands in. The substitution is recorded in outputs/week9_report.md; what
# is being demonstrated is the LoRA pipeline and a measurable gain over the base
# model, not a result specific to Phi-3.
LORA_BASE_MODEL = "distilgpt2"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_EPOCHS = 12
LORA_LR = 2e-4
LORA_BATCH = 8
LORA_MAX_LEN = 160
LORA_SEED = 0
LORA_TRAIN_FRACTION = 0.8

# ── priority routing and emissions ──────────────────────────────────────────
PRIORITY_CHART = os.path.join(OUTPUTS_DIR, "week9_priority.png")
PRIORITY_CSV = os.path.join(OUTPUTS_DIR, "week9_priority_metrics.csv")
PRIORITY_RESULTS = os.path.join(OUTPUTS_DIR, "week9_priority_results.json")

# How far ahead of a junction an emergency vehicle triggers preemption, in metres.
PREEMPT_RANGE_M = 120.0
EMERGENCY_COUNT = 12
PRIORITY_SEEDS = [0, 1, 2]
PRIORITY_SIM_SECONDS = 1800

# Weight on the emissions term in the shaped reward. Deliberately small: CO2 and
# waiting time are strongly correlated, so a large weight mostly re-weights the
# delay term and teaches nothing new.
EMISSION_WEIGHT = 0.05
