"""Simple z-score anomaly detector for SmartFlow metric rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyResult:
    """Anomaly check output."""

    anomaly: bool
    z_score: float


def z_score(value: float, baseline_values: Iterable[float]) -> float:
    """Compute a population z-score with a zero-variance guard."""
    values = list(baseline_values)
    if not values:
        raise ValueError("At least one baseline value is required.")
    sigma = pstdev(values)
    if sigma == 0:
        return 0.0
    return (value - mean(values)) / sigma


def check_anomaly(value: float, baseline_values: Iterable[float], threshold: float = 3.0) -> AnomalyResult:
    """Return whether a value is anomalous relative to the baseline distribution."""
    score = z_score(value, baseline_values)
    return AnomalyResult(anomaly=abs(score) >= threshold, z_score=round(score, 4))
