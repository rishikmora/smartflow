"""Vision service API for SmartFlow."""

from __future__ import annotations

import base64
import binascii
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI(title="SmartFlow Vision Service")


class DetectRequest(BaseModel):
    """Base64 frame detection request."""

    image_b64: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "vision_service"}


@app.post("/detect")
def detect(request: DetectRequest) -> dict[str, list[dict[str, float | str]]]:
    """Validate an image payload and return detections placeholder."""
    try:
        base64.b64decode(request.image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Malformed base64 image.") from exc
    return {"detections": []}
