"""Statistical anomaly detection over SmartFlow's live metric stream.

Two layers:

* :func:`z_score` / :func:`check_anomaly` - the original one-shot check against a
  fixed baseline, kept because it is the simplest thing that states the idea.
* :class:`StreamingDetector` - what Week 8 actually runs. It is **causal**: at
  every sample the score is computed against a rolling window of *earlier*
  samples only. A detector allowed to see the whole episode before deciding
  would flag an injected incident trivially and prove nothing.

The detector also refuses to fire until it has seen ``min_samples`` values, so a
cold start does not produce a burst of meaningless alarms, and it enforces a
refractory period so one incident is reported as one event rather than as fifty
consecutive alarms.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyResult:
    """Anomaly check output.

    Attributes:
        anomaly: whether the value breached the threshold.
        z_score: the signed score, rounded.
    """

    anomaly: bool
    z_score: float


def z_score(value: float, baseline_values: Iterable[float]) -> float:
    """Compute a population z-score with a zero-variance guard.

    Args:
        value: the observation to score.
        baseline_values: the reference distribution.

    Returns:
        The signed z-score, or 0.0 when the baseline has no spread.

    Raises:
        ValueError: if the baseline is empty.
    """
    values = list(baseline_values)
    if not values:
        raise ValueError("At least one baseline value is required.")
    sigma = pstdev(values)
    if sigma == 0:
        return 0.0
    return (value - mean(values)) / sigma


def check_anomaly(value: float, baseline_values: Iterable[float],
                  threshold: float = 3.0) -> AnomalyResult:
    """Return whether a value is anomalous relative to a baseline.

    Args:
        value: the observation to score.
        baseline_values: the reference distribution.
        threshold: absolute z above which the value is flagged.

    Returns:
        The result, carrying the score used to decide.
    """
    score = z_score(value, baseline_values)
    return AnomalyResult(anomaly=abs(score) >= threshold, z_score=round(score, 4))


@dataclass
class Alarm:
    """One raised alarm.

    Attributes:
        t: simulation time, in seconds, at which it fired.
        signal: which signal breached.
        value: the observed value.
        z: the score that triggered it.
    """

    t: float
    signal: str
    value: float
    z: float


@dataclass
class StreamingDetector:
    """Causal rolling-window z-score detector over one named signal.

    Attributes:
        signal: name of the signal being watched.
        window: how many past samples form the baseline.
        threshold: absolute z above which an alarm is raised.
        min_samples: samples required before the detector may fire at all.
        refractory_s: seconds after an alarm during which it stays quiet, so one
            incident is one event rather than a run of duplicates.
        one_sided: when True only positive excursions fire, which is what queue
            length needs - a queue that empties is not an incident.
        persistence: consecutive breaching samples required before an alarm. A
            single sample above threshold is usually the signal cycle, not an
            incident; requiring a run is the cheapest way to cut false alarms
            without desensitising the detector.
        sigma_floor: smallest standard deviation used when scoring. A lane that
            has been empty for the whole window has zero variance, and dividing
            by it would either explode or - with a naive zero guard - score every
            excursion as 0.0 and never alarm at all. The floor is in the signal's
            own units: for a queue, half a vehicle.
    """

    signal: str
    window: int = 60
    threshold: float = 3.5
    min_samples: int = 20
    refractory_s: float = 30.0
    one_sided: bool = True
    persistence: int = 1
    sigma_floor: float = 0.5

    _history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _last_alarm_t: float | None = field(default=None, init=False, repr=False)
    _run: int = field(default=0, init=False, repr=False)
    alarms: list[Alarm] = field(default_factory=list, init=False)
    scores: list[tuple[float, float]] = field(default_factory=list, init=False)

    def update(self, t: float, value: float) -> Alarm | None:
        """Score one new sample and, if it breaches, raise an alarm.

        The value is scored against the window of samples that came *before* it,
        then appended. Scoring against a window that already contains the value
        would dilute exactly the excursion the detector is meant to catch.

        Args:
            t: simulation time in seconds.
            value: the observed value.

        Returns:
            The alarm if one fired, otherwise None.
        """
        alarm: Alarm | None = None
        if len(self._history) >= self.min_samples:
            # A zero-variance window must not silence the detector: an always-empty
            # lane that suddenly queues is exactly the event worth catching.
            sigma = max(pstdev(self._history), self.sigma_floor)
            mu = mean(self._history)
            score = (value - mu) / sigma
            self.scores.append((t, score))

            breach = score >= self.threshold if self.one_sided else abs(score) >= self.threshold
            self._run = self._run + 1 if breach else 0
            quiet = (self._last_alarm_t is None
                     or (t - self._last_alarm_t) >= self.refractory_s)
            if self._run >= self.persistence and quiet:
                alarm = Alarm(t=t, signal=self.signal, value=value, z=round(score, 3))
                self.alarms.append(alarm)
                self._last_alarm_t = t
        else:
            self.scores.append((t, 0.0))

        self._history.append(value)
        while len(self._history) > self.window:
            self._history.popleft()
        return alarm


def score_detections(alarms: Iterable[Alarm],
                     incidents: Iterable[tuple[float, float]],
                     grace_s: float = 120.0) -> dict[str, float]:
    """Score alarms against known incident windows.

    An alarm counts as a true positive if it falls inside an incident window
    extended by ``grace_s``, because a queue keeps growing for a while after an
    obstruction clears and flagging that is correct, not a false alarm.

    Args:
        alarms: raised alarms.
        incidents: ``(start_s, end_s)`` windows of injected incidents.
        grace_s: seconds after an incident during which alarms still count.

    Returns:
        precision, recall, f1, detected/total incident counts, false alarms and
        mean detection latency in seconds (NaN when nothing was detected).
    """
    alarms = list(alarms)
    incidents = list(incidents)

    matched_incidents: set[int] = set()
    true_positive = 0
    latencies: list[float] = []
    for alarm in alarms:
        hit = None
        for index, (start, end) in enumerate(incidents):
            if start <= alarm.t <= end + grace_s:
                hit = index
                break
        if hit is None:
            continue
        true_positive += 1
        if hit not in matched_incidents:
            matched_incidents.add(hit)
            latencies.append(alarm.t - incidents[hit][0])

    false_positive = len(alarms) - true_positive
    precision = true_positive / len(alarms) if alarms else 0.0
    recall = len(matched_incidents) / len(incidents) if incidents else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "alarms": float(len(alarms)),
        "incidents": float(len(incidents)),
        "detected": float(len(matched_incidents)),
        "false_alarms": float(false_positive),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else float("nan"),
    }
