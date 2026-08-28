"""Render a labelled vehicle-detection dataset from the SmartFlow corridor.

The simulation runs headless under TraCI; each frame is drawn directly from the
network geometry and the live vehicle state, so every bounding box is exact
ground truth computed from the same numbers that drew the pixels. No hand
annotation, and no annotation error.

**Why not UA-DETRAC, and why not sumo-gui.** The roadmap's substitution is
"YOLOv8 trained on UA-DETRAC (public dataset) + rendered SUMO-GUI frames".
UA-DETRAC is a ~5 GB download behind a registration wall and cannot be fetched
unattended. sumo-gui's own ``gui.screenshot`` was tried first and proved
unstable here - it hung the GUI process partway through a capture run, twice -
so frames are rendered from the twin's geometry instead. That keeps the
pipeline headless, deterministic and reproducible on any machine.

What this means for the result is stated plainly in ``outputs/week8_report.md``:
the detector is tested on rendered overhead frames, not real camera footage.
There is no occlusion, no lens weather, no motion blur and no perspective. It
demonstrates a working detection pipeline end to end; it does not demonstrate
readiness for real traffic imagery.

Usage:
    python src/vision_dataset.py
    python src/vision_dataset.py --frames 8 --fresh
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import traci
from PIL import Image, ImageDraw

from week4_config import CORRIDOR_NET, CORRIDOR_ROUTE
from week8_config import (
    CAPTURE_EVERY_S,
    FRAMES_DIR,
    FRAMES_PER_VIEW,
    HELD_OUT_JUNCTIONS,
    TRAIN_JUNCTIONS,
    TRUCK_FRACTION,
    TRUCK_LENGTH_M,
    TRUCK_WIDTH_M,
    VEHICLE_CLASSES,
    VIEW_HALF_M,
    VISION_SCENARIOS,
    WARMUP_S,
    YOLO_IMAGE_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_BIN = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
ROUTES = {
    "base": CORRIDOR_ROUTE,
    "peak": os.path.join(ROOT, "data", "corridor_peak.rou.xml"),
    "light": os.path.join(ROOT, "data", "corridor_light.rou.xml"),
}

GRASS = (86, 122, 74)
ASPHALT = (58, 58, 62)
KERB = (120, 120, 124)
MARK = (208, 202, 150)
JUNCTION = (68, 68, 72)

# A single-colour fleet would let a detector win on hue alone, which would not be
# a detection result. Colours are assigned per vehicle and held fixed.
PALETTE = [
    (206, 62, 54), (226, 168, 46), (66, 132, 200), (222, 222, 226),
    (46, 46, 52), (86, 166, 108), (150, 90, 180), (188, 116, 70),
]


class View:
    """World-to-pixel transform for one square camera view.

    Attributes:
        size: image edge length in pixels.
        scale: pixels per metre.
    """

    def __init__(self, cx: float, cy: float, half_m: float, size: int) -> None:
        """Create a view centred on a world point.

        Args:
            cx: view centre x, in metres.
            cy: view centre y, in metres.
            half_m: half-width of the view, in metres.
            size: output image edge, in pixels.
        """
        self.cx, self.cy, self.half_m, self.size = cx, cy, half_m, size
        self.scale = size / (2.0 * half_m)
        self.bounds = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """Project world metres to image pixels.

        Args:
            x: world x.
            y: world y.

        Returns:
            ``(px, py)`` with y flipped, since image y grows downward.
        """
        return ((x - self.bounds[0]) * self.scale,
                (self.bounds[3] - y) * self.scale)

    def metres(self, m: float) -> float:
        """Convert a length in metres to pixels.

        Args:
            m: length in metres.

        Returns:
            The same length in pixels.
        """
        return m * self.scale


def classify(vehicle_id: str) -> int:
    """Assign a vehicle to a class deterministically from its id.

    A hash rather than a random draw keeps the dataset reproducible without
    threading a seed through TraCI.

    Args:
        vehicle_id: SUMO vehicle id.

    Returns:
        Index into :data:`VEHICLE_CLASSES`.
    """
    return 1 if (hashlib.md5(vehicle_id.encode("utf-8")).digest()[0] / 255.0) < TRUCK_FRACTION else 0


def colour_of(vehicle_id: str) -> tuple[int, int, int]:
    """Pick a stable colour for a vehicle.

    Args:
        vehicle_id: SUMO vehicle id.

    Returns:
        An RGB triple.
    """
    return PALETTE[hashlib.md5(vehicle_id.encode("utf-8")).digest()[1] % len(PALETTE)]


def draw_network(view: View, net) -> Image.Image:  # noqa: ANN001
    """Draw the static road layout for one view.

    Args:
        view: the camera view.
        net: a parsed sumolib network.

    Returns:
        An RGB image of roads and junctions with no vehicles.
    """
    image = Image.new("RGB", (view.size, view.size), GRASS)
    draw = ImageDraw.Draw(image)

    for edge in net.getEdges():
        for lane in edge.getLanes():
            shape = [view.to_px(x, y) for x, y in lane.getShape()]
            if len(shape) < 2:
                continue
            width = max(2.0, view.metres(lane.getWidth()))
            draw.line(shape, fill=KERB, width=int(width + 3), joint="curve")
            draw.line(shape, fill=ASPHALT, width=int(width), joint="curve")

    for node in net.getNodes():
        shape = node.getShape()
        if shape and len(shape) >= 3:
            draw.polygon([view.to_px(x, y) for x, y in shape], fill=JUNCTION)

    # centre dashes, drawn after junction pads so they are not covered
    for edge in net.getEdges():
        for lane in edge.getLanes():
            pts = lane.getShape()
            if len(pts) < 2:
                continue
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 12:
                continue
            steps = int(length // 8)
            for i in range(0, steps, 2):
                a = i / max(steps, 1)
                b = min((i + 1) / max(steps, 1), 1.0)
                p0 = view.to_px(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
                p1 = view.to_px(x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)
                draw.line([p0, p1], fill=MARK, width=1)
    return image


def vehicle_corners(x: float, y: float, angle_deg: float,
                    length: float, width: float) -> list[tuple[float, float]]:
    """Return a vehicle's four world-space corners.

    ``getPosition`` gives the front bumper, so the centre is half a length back
    along the heading. SUMO angles are degrees clockwise from north.

    Args:
        x: front-bumper x, in metres.
        y: front-bumper y, in metres.
        angle_deg: heading, degrees clockwise from north.
        length: vehicle length in metres.
        width: vehicle width in metres.

    Returns:
        Four ``(x, y)`` corners in metres.
    """
    rad = math.radians(angle_deg)
    hx, hy = math.sin(rad), math.cos(rad)
    px, py = math.cos(rad), -math.sin(rad)
    cx, cy = x - hx * length / 2.0, y - hy * length / 2.0
    corners = []
    for along, across in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        corners.append((cx + hx * along * length / 2.0 + px * across * width / 2.0,
                        cy + hy * along * length / 2.0 + py * across * width / 2.0))
    return corners


def to_yolo(corners_px: list[tuple[float, float]], size: int
            ) -> tuple[float, float, float, float] | None:
    """Convert projected corners to a normalised YOLO box.

    Args:
        corners_px: the four corners in pixels.
        size: image edge in pixels.

    Returns:
        ``(cx, cy, w, h)`` normalised to 0-1, or None if it falls outside or
        clips to nothing.
    """
    xs = [c[0] for c in corners_px]
    ys = [c[1] for c in corners_px]
    x0, x1 = max(0.0, min(xs)), min(float(size), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(size), max(ys))
    w, h = x1 - x0, y1 - y0
    if w <= 1.5 or h <= 1.5:
        return None
    return ((x0 + w / 2) / size, (y0 + h / 2) / size, w / size, h / size)


def add_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Add mild sensor-like noise and exposure jitter.

    Without this every frame is a pixel-exact render and the detector can latch
    onto rendering artefacts rather than shape. It does not make the task
    realistic - it only stops it being degenerate.

    Args:
        image: the clean render.
        rng: seeded random generator.

    Returns:
        The perturbed image.
    """
    array = np.asarray(image).astype(np.float32)
    array *= rng.uniform(0.88, 1.12)                       # exposure
    array += rng.normal(0.0, 4.5, array.shape)             # sensor noise
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def capture_view(junction: str, scenario: str, split: str, frames: int,
                 net, seed: int) -> int:  # noqa: ANN001
    """Render and label one junction under one demand scenario.

    Args:
        junction: junction id to centre on.
        scenario: key into :data:`ROUTES`.
        split: ``"train"`` or ``"val"``.
        frames: frames to capture.
        net: parsed sumolib network.
        seed: seed for the noise generator.

    Returns:
        Frames written.

    Raises:
        FileNotFoundError: if the scenario route file is missing.
    """
    route = ROUTES[scenario]
    if not os.path.isfile(route):
        raise FileNotFoundError(f"Route file missing: {route}")

    node = net.getNode(junction)
    cx, cy = node.getCoord()
    view = View(cx, cy, VIEW_HALF_M, YOLO_IMAGE_SIZE)
    backdrop = draw_network(view, net)

    images = os.path.join(FRAMES_DIR, "images", split)
    labels = os.path.join(FRAMES_DIR, "labels", split)
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    rng = np.random.default_rng(seed)

    traci.start([SUMO_BIN, "-n", CORRIDOR_NET, "-r", route,
                 "--no-step-log", "--no-warnings", "--time-to-teleport", "-1"])
    written = 0
    try:
        for _ in range(WARMUP_S):
            traci.simulationStep()

        for index in range(frames):
            for _ in range(CAPTURE_EVERY_S):
                traci.simulationStep()

            frame = backdrop.copy()
            draw = ImageDraw.Draw(frame)
            rows: list[str] = []

            for vid in traci.vehicle.getIDList():
                try:
                    x, y = traci.vehicle.getPosition(vid)
                except traci.TraCIException:
                    continue
                if not (view.bounds[0] - 15 <= x <= view.bounds[2] + 15
                        and view.bounds[1] - 15 <= y <= view.bounds[3] + 15):
                    continue
                cls = classify(vid)
                length = TRUCK_LENGTH_M if cls else traci.vehicle.getLength(vid)
                width = TRUCK_WIDTH_M if cls else traci.vehicle.getWidth(vid)
                angle = traci.vehicle.getAngle(vid)

                corners = vehicle_corners(x, y, angle, length, width)
                corners_px = [view.to_px(*c) for c in corners]
                draw.polygon(corners_px, fill=colour_of(vid), outline=(20, 20, 22))

                box = to_yolo(corners_px, view.size)
                if box is not None:
                    rows.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")

            stem = f"{junction}_{scenario}_{index:03d}"
            add_noise(frame, rng).save(os.path.join(images, stem + ".png"))
            with open(os.path.join(labels, stem + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(rows))
            written += 1
    finally:
        traci.close()

    log.info("  %-3s %-6s %-5s -> %3d frames", junction, scenario, split, written)
    return written


def write_data_yaml() -> str:
    """Write the ultralytics dataset descriptor.

    Returns:
        Path to data.yaml.
    """
    path = os.path.join(FRAMES_DIR, "data.yaml")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(VEHICLE_CLASSES))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"path: {FRAMES_DIR.replace(os.sep, '/')}\n"
                     "train: images/train\n"
                     "val: images/val\n"
                     "names:\n" + names + "\n")
    return path


def main() -> None:
    """Render the whole dataset."""
    parser = argparse.ArgumentParser(description="Render a labelled detection dataset.")
    parser.add_argument("--frames", type=int, default=FRAMES_PER_VIEW)
    parser.add_argument("--fresh", action="store_true", help="Delete any existing dataset.")
    args = parser.parse_args()

    if args.fresh and os.path.isdir(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    import sumolib
    net = sumolib.net.readNet(CORRIDOR_NET)

    plan = []
    for scenario in VISION_SCENARIOS:
        plan += [(j, scenario, "train") for j in TRAIN_JUNCTIONS]
        plan += [(j, scenario, "val") for j in HELD_OUT_JUNCTIONS]

    log.info("Rendering %d views x %d frames at %dpx",
             len(plan), args.frames, YOLO_IMAGE_SIZE)
    total = 0
    for seed, (junction, scenario, split) in enumerate(plan):
        total += capture_view(junction, scenario, split, args.frames, net, seed)

    yaml_path = write_data_yaml()
    counts = {}
    for split in ("train", "val"):
        directory = os.path.join(FRAMES_DIR, "labels", split)
        boxes = 0
        if os.path.isdir(directory):
            for name in os.listdir(directory):
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    boxes += sum(1 for line in handle if line.strip())
        counts[split] = boxes

    log.info("Wrote %d frames; boxes: train=%d val=%d",
             total, counts.get("train", 0), counts.get("val", 0))
    log.info("Descriptor: %s", yaml_path)
    log.info("Held out for validation: junction %s", ", ".join(HELD_OUT_JUNCTIONS))


if __name__ == "__main__":
    main()
