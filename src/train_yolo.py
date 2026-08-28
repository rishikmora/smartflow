"""Train and evaluate a YOLOv8 vehicle detector on the rendered corridor frames.

Trains on frames from junctions B1, B2 and C1 and validates on C2, which the
model never sees during training. Holding out a whole junction rather than a
random sample of frames is the only split that means anything here: consecutive
frames of one junction are near-duplicates, so a random split would leak and
report a score that says nothing about generalisation.

Runs on CPU. There is no CUDA device on this machine, which is why the model is
the smallest in the family (``yolov8n``) and the image budget is modest.

Usage:
    python src/train_yolo.py
    python src/train_yolo.py --epochs 5        # quick pipeline check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week8_config import (
    FRAMES_DIR,
    VISION_DIR,
    VISION_MODEL_DIR,
    YOLO_BATCH,
    YOLO_EPOCHS,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
    YOLO_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

RESULTS_JSON = os.path.join(VISION_DIR, "detector_results.json")


def dataset_summary() -> dict[str, int]:
    """Count frames and boxes per split.

    Returns:
        Mapping like ``{"train_images": .., "train_boxes": .., ...}``.

    Raises:
        FileNotFoundError: if the dataset has not been rendered.
    """
    if not os.path.isfile(os.path.join(FRAMES_DIR, "data.yaml")):
        raise FileNotFoundError(
            f"No dataset at {FRAMES_DIR}. Run: python src/vision_dataset.py"
        )
    out: dict[str, int] = {}
    for split in ("train", "val"):
        images = os.path.join(FRAMES_DIR, "images", split)
        labels = os.path.join(FRAMES_DIR, "labels", split)
        out[f"{split}_images"] = (
            len([n for n in os.listdir(images) if n.endswith(".png")])
            if os.path.isdir(images) else 0
        )
        boxes = 0
        classes: dict[str, int] = {}
        if os.path.isdir(labels):
            for name in os.listdir(labels):
                with open(os.path.join(labels, name), encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            boxes += 1
                            key = line.split()[0]
                            classes[key] = classes.get(key, 0) + 1
        out[f"{split}_boxes"] = boxes
        out[f"{split}_cars"] = classes.get("0", 0)
        out[f"{split}_trucks"] = classes.get("1", 0)
    return out


def train(epochs: int) -> tuple[object, dict[str, float]]:
    """Train the detector and validate it on the held-out junction.

    Args:
        epochs: training epochs.

    Returns:
        ``(model, metrics)`` where metrics carries mAP50, mAP50-95, precision
        and recall on the held-out split.
    """
    from ultralytics import YOLO

    os.makedirs(VISION_MODEL_DIR, exist_ok=True)
    model = YOLO(YOLO_MODEL)
    model.train(
        data=os.path.join(FRAMES_DIR, "data.yaml"),
        epochs=epochs,
        imgsz=YOLO_IMAGE_SIZE,
        batch=YOLO_BATCH,
        seed=YOLO_SEED,
        device="cpu",
        project=VISION_MODEL_DIR,
        name="corridor",
        exist_ok=True,
        verbose=False,
        plots=True,
    )
    result = model.val(
        data=os.path.join(FRAMES_DIR, "data.yaml"),
        imgsz=YOLO_IMAGE_SIZE,
        device="cpu",
        project=VISION_MODEL_DIR,
        name="corridor_val",
        exist_ok=True,
        verbose=False,
    )
    box = result.box
    metrics = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    for index, name in enumerate(("car", "truck")):
        try:
            metrics[f"mAP50_{name}"] = float(box.ap50[index])
        except (IndexError, TypeError):
            pass
    return model, metrics


def main() -> None:
    """Train, evaluate and record the detector."""
    parser = argparse.ArgumentParser(description="Train the corridor vehicle detector.")
    parser.add_argument("--epochs", type=int, default=YOLO_EPOCHS)
    args = parser.parse_args()

    summary = dataset_summary()
    log.info("Dataset: train %d images / %d boxes | val %d images / %d boxes",
             summary["train_images"], summary["train_boxes"],
             summary["val_images"], summary["val_boxes"])
    if summary["val_images"] == 0:
        raise SystemExit("Validation split is empty; re-render the dataset.")

    log.info("Training %s for %d epochs on CPU (this takes a while)...",
             YOLO_MODEL, args.epochs)
    _model, metrics = train(args.epochs)

    log.info("Held-out junction results:")
    for key, value in metrics.items():
        log.info("  %-14s %.4f", key, value)

    payload = {"dataset": summary, "metrics": metrics, "epochs": args.epochs,
               "model": YOLO_MODEL, "device": "cpu"}
    os.makedirs(VISION_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("Wrote %s", RESULTS_JSON)

    weights = os.path.join(VISION_MODEL_DIR, "corridor", "weights", "best.pt")
    if os.path.isfile(weights):
        log.info("Weights: %s", weights)


if __name__ == "__main__":
    main()
