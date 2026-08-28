# Week 8 — Deferred and Substituted Items

All three Week 8 components ran and met their Definition of Done. Two
substitutions were made along the way, both recorded here rather than folded
quietly into the result.

## Substituted

| Roadmap item | What was done instead | Why |
|---|---|---|
| YOLOv8 trained on **UA-DETRAC** + rendered SUMO-GUI frames | Trained on rendered frames only | UA-DETRAC is a ~5 GB download behind a registration wall and cannot be fetched unattended |
| Frames captured via **sumo-gui screenshots** | Frames rendered headlessly from network geometry + live TraCI state | `gui.screenshot` hung the sumo-gui process partway through a capture run, twice, leaving orphaned processes. The headless renderer is deterministic, needs no display, and produces exact ground-truth boxes by construction |

**What this costs the result.** The detector is tested on rendered overhead
frames: no occlusion, no perspective, no motion blur, no weather on the lens,
clean background. mAP50 ≈ 0.99 on a held-out junction demonstrates that the
pipeline works end to end — dataset generation, labelling, training, held-out
evaluation. It does **not** demonstrate readiness for real traffic camera
footage, and no real-camera number has been measured by this project.

Closing this properly needs someone to download UA-DETRAC manually and re-run
`src/train_yolo.py` against a mixed dataset. The training code needs no change;
only `data.yaml` would point somewhere else.

## Genuinely deferred

- **Vision in the live loop.** The detector runs offline against recorded
  frames. Nothing feeds its output back into signal control, and nothing should
  until it is tested on real imagery.
- **Emission and noise models beyond CO2.** SUMO exposes CO, HC, NOx, PMx and
  noise per lane; only CO2 is collected.

## Not deferred, contrary to the earlier version of this file

The previous version of this note listed a z-score detector whose "dataset-level
false positive/negative rates are not yet measured". They are now: the detector
is scored against injected incidents with exact ground truth across three seeds,
and the whole precision/recall/latency trade-off is swept rather than reported at
one flattering operating point. See `week8_report.md` and
`week8_anomaly_results.json`.
