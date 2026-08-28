"""Vision service: the Week 8 detector and the incident detector's findings.

Reports what the perception side measured — the detector's held-out scores and
the anomaly detector's operating point — and, when the trained weights are on
the mount, runs the detector over an uploaded frame.

Inference is offered because it is cheap and bounded. Training is not: it takes
about an hour on CPU and belongs in the harness.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from common import data, events, settings
from common.auth import auth_mode, require_user
from common.observability import instrument

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

SERVICE = "vision_service"
WEIGHTS = os.path.join(settings.DATA_ROOT, "models", "yolo", "corridor",
                       "weights", "best.pt")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = FastAPI(
    title="SmartFlow Vision Service",
    description="Vehicle detection and incident detection results. Read-only.",
    version="1.0.0",
)
instrument(app, SERVICE)

_model: Any = None


def _load_model() -> Any:
    """Load the trained detector once, on first use.

    Returns:
        A YOLO model.

    Raises:
        HTTPException: if the weights or ultralytics are unavailable.
    """
    global _model
    if _model is not None:
        return _model
    if not os.path.isfile(WEIGHTS):
        raise HTTPException(
            status_code=503,
            detail=f"Detector weights not mounted at {WEIGHTS}. "
                   "Run src/train_yolo.py, or mount models/.",
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="ultralytics is not installed in this image; /detect is unavailable.",
        ) from exc
    _model = YOLO(WEIGHTS)
    log.info("Loaded detector from %s", WEIGHTS)
    return _model


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Report service health, data availability and whether inference is possible.

    Returns:
        Status and capability detail.
    """
    available = data.data_available()
    return {
        "status": "ok" if available["detector_metrics"] else "degraded",
        "service": SERVICE,
        "auth": auth_mode(),
        "events": events.broker_status(),
        "weights_present": os.path.isfile(WEIGHTS),
        "data": available,
    }


@app.get("/detector", tags=["vision"])
def detector(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """Return the detector's held-out evaluation.

    Args:
        _user: authenticated principal.

    Returns:
        Dataset composition and held-out metrics, with the caveat attached.

    Raises:
        HTTPException: if the detector has not been trained.
    """
    payload = data.detector_metrics()
    if payload is None:
        raise HTTPException(status_code=404,
                            detail="Detector metrics not present. Run src/train_yolo.py.")
    return {
        **payload,
        "caveat": "Measured on frames rendered from the simulation - no occlusion, "
                  "perspective, motion blur or lens weather. Not a real-camera result.",
    }


@app.get("/anomalies", tags=["vision"])
def anomalies(_user: dict = Depends(require_user)) -> dict[str, Any]:
    """Return the incident detector's operating point and sweep.

    Args:
        _user: authenticated principal.

    Returns:
        The chosen operating point plus the full sweep.

    Raises:
        HTTPException: if the experiment has not been run.
    """
    payload = data.anomaly_results()
    if payload is None:
        raise HTTPException(status_code=404,
                            detail="Anomaly results not present. Run src/anomaly_live.py.")
    return {
        "incidents": payload["incidents"],
        "blocked_edges": payload["blocked_edges"],
        "chosen": payload["chosen"],
        "sweep_points": len(payload["sweep"]),
        "sweep": payload["sweep"],
    }


@app.post("/detect", tags=["vision"])
async def detect(
    frame: UploadFile = File(..., description="A PNG or JPEG frame"),
    _user: dict = Depends(require_user),
) -> dict[str, Any]:
    """Run the trained detector over one uploaded frame.

    Args:
        frame: the uploaded image.
        _user: authenticated principal.

    Returns:
        Detected boxes in pixel coordinates with class names and confidences.

    Raises:
        HTTPException: if the upload is too large, unreadable, or the model is
            unavailable.
    """
    payload = await frame.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"Frame exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - untrusted upload
        raise HTTPException(status_code=400,
                            detail=f"Could not read the image: {exc}") from exc

    model = _load_model()
    results = model.predict(image, device="cpu", verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append({
                "class": names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "box": {"x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1)},
            })
    events.publish(SERVICE, "frame.detected", {"detections": len(detections)})
    return {"width": image.width, "height": image.height,
            "count": len(detections), "detections": detections}


@app.on_event("shutdown")
def _shutdown() -> None:
    """Flush buffered analytics events before the process exits."""
    events.close()
