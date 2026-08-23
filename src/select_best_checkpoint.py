"""Pick each Week 3 seed's policy by validation score instead of taking the last iterate.

Why this exists
---------------
PPO's final weights are not its best weights. On this corridor the training returns for
seed 1 swing between -1.3 and -959 across consecutive episodes: near saturation, one
unlucky action sequence cascades into a jam that the rest of the episode cannot clear.
Reporting whichever policy happened to exist when the step budget ran out makes the
result a lottery, and seed 1 lost it.

Selecting the best checkpoint is standard practice, but only if the selection is made on
data that is *not* the reported result. So:

* candidates — every intermediate checkpoint plus the final model;
* selection score — one deterministic episode on a **validation seed**
  (``VALIDATION_SEED_OFFSET + seed``) that was never used for training or for the
  reported evaluation;
* reported result — a separate evaluation on seeds 0/1/2, run afterwards by
  ``eval_corridor.py``.

Picking the checkpoint that scores best on seeds 0/1/2 directly would be tuning on the
test set, and the reported improvement would be meaningless.

Usage:
    python src/select_best_checkpoint.py
    python src/select_best_checkpoint.py --seeds 1 --metric avg_wait_time_s

Writes the chosen model to ``models/ppo_corridor_seed{N}.zip`` (backing up the final
iterate as ``..._final.zip``) and records the full selection table in
``outputs/week3_checkpoint_selection.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_corridor import eval_ppo
from week3_config import MODELS_DIR, OUTPUTS_DIR, TRAIN_SEEDS, model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_BASE = os.path.join(OUTPUTS_DIR, "checkpoints")
SELECTION_JSON = os.path.join(OUTPUTS_DIR, "week3_checkpoint_selection.json")

# Validation seeds must not collide with the reported evaluation seeds (0, 1, 2).
VALIDATION_SEED_OFFSET = 100

# Lower is better for these; higher is better otherwise.
LOWER_IS_BETTER = {"avg_wait_time_s", "max_queue_len"}


def candidates(seed: int) -> list[tuple[str, str]]:
    """List the checkpoints available for one seed, oldest first.

    Args:
        seed: training seed.

    Returns:
        ``[(label, path)]`` including the final model as ``"final"``.
    """
    pattern = os.path.join(CHECKPOINT_BASE, f"ppo_corridor_seed{seed}", f"ppo_corridor_seed{seed}_*.zip")
    found: list[tuple[int, str]] = []
    for path in glob.glob(pattern):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    out = [(f"{steps}_steps", path) for steps, path in found]

    final = model_path(seed)
    if os.path.isfile(final):
        out.append(("final", final))
    return out


def score(path: str, validation_seed: int, metric: str) -> dict[str, Any]:
    """Score one checkpoint with a deterministic episode on the validation seed.

    Args:
        path: checkpoint ``.zip``.
        validation_seed: SUMO seed for the validation episode.
        metric: junction-scope metric to select on.

    Returns:
        The junction-scope metric row, plus the selection ``score``.
    """
    rows = eval_ppo(validation_seed, model_file=path, controller_label="ppo_validation")
    junction = next(row for row in rows if row["scope"] == "junction")
    corridor = next(row for row in rows if row["scope"] == "corridor")
    return {
        "junction": junction,
        "corridor_avg_wait_time_s": corridor["avg_wait_time_s"],
        "score": float(junction[metric]),
    }


def _label_steps(label: str) -> float:
    """Return the training-step count a candidate label represents.

    Args:
        label: ``"25000_steps"`` or ``"final"``.

    Returns:
        The step count; ``inf`` for the final iterate, which is the most-trained.
    """
    if label == "final":
        return float("inf")
    match = re.match(r"(\d+)_steps$", label)
    return float(match.group(1)) if match else 0.0


def pick_best(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Choose the winning candidate, breaking ties toward the more-trained policy.

    Several checkpoints often score *identically* here: once the policy converges it
    behaves the same way on the validation episode, so its score stops moving. Breaking
    those ties by list order would report a 25k-step policy from a 200k-step run for no
    reason other than that it was scored first. Among equal scores the most-trained
    checkpoint is the natural choice.

    Args:
        results: candidate records, each with ``label`` and ``score``.
        metric: the metric being selected on, which sets the direction.

    Returns:
        The winning candidate record.

    Raises:
        ValueError: if ``results`` is empty.
    """
    if not results:
        raise ValueError("No candidates to choose from.")
    sign = 1.0 if metric in LOWER_IS_BETTER else -1.0
    # primary: best score; tiebreak: most training steps (negated for ascending sort)
    return sorted(results, key=lambda item: (sign * item["score"], -_label_steps(item["label"])))[0]


def select(seed: int, metric: str = "avg_wait_time_s") -> dict[str, Any]:
    """Choose the best checkpoint for one seed and install it as that seed's model.

    Args:
        seed: training seed.
        metric: junction-scope metric to select on.

    Returns:
        A record of every candidate's score and which was chosen.

    Raises:
        FileNotFoundError: if the seed has no checkpoints at all.
    """
    options = candidates(seed)
    if not options:
        raise FileNotFoundError(
            f"No checkpoints or final model found for seed {seed}. "
            "Run 'python src/train_ppo_corridor.py --seed {seed}' first."
        )

    validation_seed = VALIDATION_SEED_OFFSET + seed
    results: list[dict[str, Any]] = []
    for label, path in options:
        measured = score(path, validation_seed, metric)
        results.append({"label": label, "path": path, **measured})
        log.info(
            "seed=%d validation_seed=%d %-14s junction %s=%.2f (queue=%s, throughput=%s)",
            seed, validation_seed, label, metric, measured["score"],
            measured["junction"]["max_queue_len"], measured["junction"]["throughput_veh"],
        )

    best = pick_best(results, metric)

    final_path = model_path(seed)
    backup = final_path.replace(".zip", "_final.zip")
    if os.path.isfile(final_path) and not os.path.isfile(backup):
        shutil.copy2(final_path, backup)
    if best["path"] != final_path:
        shutil.copy2(best["path"], final_path)

    log.info("seed=%d selected %s (%s=%.2f on validation seed %d)",
             seed, best["label"], metric, best["score"], validation_seed)
    return {
        "seed": seed,
        "validation_seed": validation_seed,
        "metric": metric,
        "selected": best["label"],
        "selected_score": best["score"],
        "candidates": results,
    }


def reselect_from_json(seed: int, metric: str = "avg_wait_time_s") -> dict[str, Any]:
    """Re-apply the selection rule to scores already recorded, without re-running SUMO.

    Validation scoring is the expensive part (one episode per candidate). When only the
    *rule* changes — a different metric, or a different tie-break — the recorded scores
    are still valid and can simply be re-ranked.

    Args:
        seed: training seed.
        metric: metric to select on; must already be present in the stored records.

    Returns:
        The updated selection record.

    Raises:
        FileNotFoundError: if no selection JSON exists.
        KeyError: if the seed was never scored.
    """
    if not os.path.isfile(SELECTION_JSON):
        raise FileNotFoundError(
            f"{SELECTION_JSON} not found — run the full selection at least once first."
        )
    with open(SELECTION_JSON, encoding="utf-8") as handle:
        stored = json.load(handle)
    record = next((r for r in stored.get("selections", []) if r["seed"] == seed), None)
    if record is None:
        raise KeyError(f"Seed {seed} has no recorded selection; run the full selection for it.")

    results = record["candidates"]
    if metric != record.get("metric"):
        for item in results:
            item["score"] = float(item["junction"][metric])
    best = pick_best(results, metric)

    final_path = model_path(seed)
    backup = final_path.replace(".zip", "_final.zip")
    if os.path.isfile(final_path) and not os.path.isfile(backup):
        shutil.copy2(final_path, backup)
    source = backup if best["label"] == "final" and os.path.isfile(backup) else best["path"]
    if os.path.abspath(source) != os.path.abspath(final_path):
        shutil.copy2(source, final_path)

    log.info("seed=%d re-selected %s (%s=%.2f, from recorded scores)",
             seed, best["label"], metric, best["score"])
    record.update({"metric": metric, "selected": best["label"], "selected_score": best["score"]})
    return record


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Select Week 3 policies by validation score.")
    parser.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--metric", default="avg_wait_time_s",
                        choices=["avg_wait_time_s", "max_queue_len", "throughput_veh"])
    parser.add_argument("--reselect", action="store_true",
                        help="Re-rank already-recorded scores instead of re-running episodes.")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    chooser = reselect_from_json if args.reselect else select
    records = [chooser(seed, args.metric) for seed in args.seeds]

    existing: list[dict[str, Any]] = []
    if os.path.isfile(SELECTION_JSON):
        try:
            with open(SELECTION_JSON, encoding="utf-8") as handle:
                existing = json.load(handle).get("selections", [])
        except (OSError, json.JSONDecodeError):
            existing = []
    merged = {record["seed"]: record for record in existing}
    merged.update({record["seed"]: record for record in records})

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(SELECTION_JSON, "w", encoding="utf-8") as handle:
        json.dump({
            "protocol": (
                "Each candidate checkpoint is scored on one deterministic episode using a "
                "validation SUMO seed (100 + training seed) that is disjoint from the "
                "reported evaluation seeds 0/1/2."
            ),
            "selections": [merged[key] for key in sorted(merged)],
        }, handle, indent=2)
    log.info("Wrote %s", SELECTION_JSON)


if __name__ == "__main__":
    main()
