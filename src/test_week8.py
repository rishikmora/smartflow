"""Self-checks for Week 8: vision labels, incident detection, scenario planning.

Each of Week 8's three Definitions of Done rests on something that could be
quietly wrong in a way the headline number would hide, so each gets a check
aimed at exactly that:

* **The detector's labels must match the pixels.** A sign error or a flipped
  axis in the world-to-image projection produces a dataset that trains happily
  and reports a high mAP against its own wrong boxes. The projection is checked
  against hand-computed values.
* **The train/test split must not leak.** Frames of one junction are
  near-duplicates, so the held-out junction must appear in validation only.
* **The anomaly detector must be causal.** A detector that scores a sample
  against a window containing that sample - or against the whole episode - would
  flag injected incidents trivially and prove nothing.

Runs standalone and is pytest-compatible.

Usage:
    python src/test_week8.py
"""

from __future__ import annotations

import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def test_world_to_pixel_projection_is_correct() -> None:
    """The view transform must place world points where they belong in pixels."""
    from vision_dataset import View

    view = View(cx=100.0, cy=200.0, half_m=50.0, size=640)

    centre = view.to_px(100.0, 200.0)
    assert abs(centre[0] - 320.0) < 1e-6, centre
    assert abs(centre[1] - 320.0) < 1e-6, centre

    # World y grows upward, image y grows downward: the top-left corner of the
    # view is (xmin, ymax), not (xmin, ymin).
    top_left = view.to_px(50.0, 250.0)
    assert abs(top_left[0]) < 1e-6 and abs(top_left[1]) < 1e-6, top_left

    bottom_right = view.to_px(150.0, 150.0)
    assert abs(bottom_right[0] - 640.0) < 1e-6, bottom_right
    assert abs(bottom_right[1] - 640.0) < 1e-6, bottom_right


def test_vehicle_corners_use_front_bumper_and_heading() -> None:
    """getPosition is the front bumper, so the body extends backwards."""
    from vision_dataset import vehicle_corners

    # Heading 90 degrees clockwise from north = due east (+x).
    corners = vehicle_corners(x=10.0, y=0.0, angle_deg=90.0, length=4.0, width=2.0)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    assert abs(max(xs) - 10.0) < 1e-6, f"front should sit at x=10, got {max(xs)}"
    assert abs(min(xs) - 6.0) < 1e-6, f"rear should sit at x=6, got {min(xs)}"
    assert abs(max(ys) - 1.0) < 1e-6 and abs(min(ys) + 1.0) < 1e-6, (xs, ys)


def test_yolo_boxes_are_normalised_and_inside_the_frame() -> None:
    """Every emitted box must be a valid normalised YOLO box."""
    from vision_dataset import to_yolo

    box = to_yolo([(100.0, 100.0), (140.0, 100.0), (140.0, 120.0), (100.0, 120.0)], 640)
    assert box is not None
    cx, cy, w, h = box
    assert abs(cx - 120.0 / 640) < 1e-6 and abs(cy - 110.0 / 640) < 1e-6, box
    assert abs(w - 40.0 / 640) < 1e-6 and abs(h - 20.0 / 640) < 1e-6, box
    for value in box:
        assert 0.0 <= value <= 1.0, box

    # A vehicle entirely outside the frame must produce no label at all.
    assert to_yolo([(-50.0, -50.0), (-40.0, -50.0), (-40.0, -40.0), (-50.0, -40.0)],
                   640) is None


def test_dataset_split_holds_out_whole_junctions() -> None:
    """The held-out junction must never appear in the training split."""
    from week8_config import FRAMES_DIR, HELD_OUT_JUNCTIONS, TRAIN_JUNCTIONS

    train_dir = os.path.join(FRAMES_DIR, "images", "train")
    val_dir = os.path.join(FRAMES_DIR, "images", "val")
    if not os.path.isdir(train_dir):
        log.warning("  dataset not rendered; skipping split check")
        return

    train_names = os.listdir(train_dir)
    val_names = os.listdir(val_dir)
    assert train_names and val_names, "both splits must be populated"

    for held_out in HELD_OUT_JUNCTIONS:
        leaked = [n for n in train_names if n.startswith(held_out + "_")]
        assert not leaked, f"{held_out} leaked into training: {leaked[:3]}"
    for name in val_names:
        junction = name.split("_")[0]
        assert junction in HELD_OUT_JUNCTIONS, f"unexpected junction in val: {name}"
    for name in train_names:
        assert name.split("_")[0] in TRAIN_JUNCTIONS, f"unexpected junction: {name}"


def test_labels_match_their_images() -> None:
    """Every rendered frame must have a label file, and boxes must be in range."""
    from week8_config import FRAMES_DIR, VEHICLE_CLASSES

    images = os.path.join(FRAMES_DIR, "images", "val")
    labels = os.path.join(FRAMES_DIR, "labels", "val")
    if not os.path.isdir(images):
        log.warning("  dataset not rendered; skipping label check")
        return

    checked = 0
    for name in sorted(os.listdir(images))[:20]:
        label_path = os.path.join(labels, name.replace(".png", ".txt"))
        assert os.path.isfile(label_path), f"missing label for {name}"
        with open(label_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, line
                assert 0 <= int(parts[0]) < len(VEHICLE_CLASSES), line
                for value in map(float, parts[1:]):
                    assert 0.0 <= value <= 1.0, line
                checked += 1
    assert checked > 0, "no boxes found in the validation labels"


def test_detector_is_causal() -> None:
    """A sample must be scored against earlier samples only."""
    from anomaly_detector import StreamingDetector

    detector = StreamingDetector(signal="q", window=20, threshold=3.0,
                                 min_samples=10, refractory_s=0.0, persistence=1)
    # A flat baseline, then one large spike.
    for t in range(10):
        assert detector.update(float(t), 5.0) is None, "fired before min_samples"
    for t in range(10, 30):
        detector.update(float(t), 5.0)

    alarm = detector.update(30.0, 500.0)
    assert alarm is not None, "a 100x excursion should fire"
    assert alarm.z > 3.0, alarm

    # If the detector had included the spike in its own baseline, the score would
    # have been diluted well below this.
    assert alarm.z > 10.0, f"score {alarm.z} suggests the sample polluted its baseline"


def test_detector_persistence_suppresses_single_spikes() -> None:
    """The persistence gate must be what decides, on identical input.

    Compared against an otherwise identical detector with persistence=1, so the
    test isolates the gate rather than the rolling baseline: a sustained
    excursion legitimately pulls the baseline up after a few samples, which is
    the detector adapting, not a fault.
    """
    from anomaly_detector import StreamingDetector

    def run(persistence: int) -> int:
        detector = StreamingDetector(signal="q", window=30, threshold=3.0,
                                     min_samples=10, refractory_s=0.0,
                                     persistence=persistence)
        for t in range(30):
            detector.update(float(t), 5.0)
        detector.update(30.0, 500.0)
        return len(detector.alarms)

    assert run(1) == 1, "persistence=1 should alarm on the first breach"
    assert run(5) == 0, "persistence=5 should not alarm on a single breach"


def test_detection_scoring_matches_hand_computation() -> None:
    """Precision, recall and latency must be computed the obvious way."""
    from anomaly_detector import Alarm, score_detections

    incidents = [(100.0, 200.0), (500.0, 600.0)]
    alarms = [
        Alarm(t=130.0, signal="a", value=1.0, z=4.0),   # inside incident 1
        Alarm(t=140.0, signal="a", value=1.0, z=4.0),   # also incident 1
        Alarm(t=900.0, signal="a", value=1.0, z=4.0),   # false alarm
    ]
    score = score_detections(alarms, incidents, grace_s=60.0)

    assert score["detected"] == 1.0, score
    assert score["false_alarms"] == 1.0, score
    assert abs(score["precision"] - 2 / 3) < 1e-9, score
    assert abs(score["recall"] - 0.5) < 1e-9, score
    assert abs(score["mean_latency_s"] - 30.0) < 1e-9, score


def test_weather_profiles_are_monotonic() -> None:
    """Fog must be modelled as at least as severe as rain, and rain than clear."""
    from week8_config import WEATHER

    clear, rain, fog = WEATHER["clear"], WEATHER["rain"], WEATHER["fog"]
    assert clear["speed_factor"] > rain["speed_factor"] > fog["speed_factor"], WEATHER
    assert clear["tau"] < rain["tau"] < fog["tau"], WEATHER
    assert clear["min_gap"] < rain["min_gap"] < fog["min_gap"], WEATHER
    assert clear["decel"] > rain["decel"] > fog["decel"], WEATHER


def test_scenario_results_are_internally_consistent() -> None:
    """Restricting a link must not improve the corridor it restricts."""
    import csv

    from week8_config import PLANNER_CSV

    if not os.path.isfile(PLANNER_CSV):
        log.warning("  scenario planner not run; skipping")
        return

    with open(PLANNER_CSV, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "planner CSV is empty"

    def mean(closure: str, weather: str, key: str) -> float:
        values = [float(r[key]) for r in rows
                  if r["closure"] == closure and r["weather"] == weather]
        return sum(values) / len(values) if values else math.nan

    base_wait = mean("none", "clear", "avg_wait_time_s")
    works_wait = mean("roadworks", "clear", "avg_wait_time_s")
    assert works_wait > base_wait, (
        f"roadworks ({works_wait:.2f}s) should not beat the open corridor "
        f"({base_wait:.2f}s)")

    base_thru = mean("none", "clear", "throughput_veh")
    rain_thru = mean("none", "rain", "throughput_veh")
    assert rain_thru < base_thru, (rain_thru, base_thru)


def main() -> None:
    """Run every check and report a pass/fail summary."""
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures: list[str] = []
    for test in tests:
        try:
            test()
            log.info("PASS  %s", test.__name__)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            failures.append(f"{test.__name__}: {exc}")
            log.error("FAIL  %s -> %s", test.__name__, exc)
    log.info("%d/%d checks passed", len(tests) - len(failures), len(tests))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
