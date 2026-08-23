"""Shared per-second metric collection for every SmartFlow controller.

Every controller from Week 1 onward is judged by the same four numbers, sampled the
same way, so results stay comparable across weeks:

``avg_wait_time_s``
    Mean over *completed* trips of the number of simulated seconds the vehicle spent
    with ``getWaitingTime() > 0``.
``max_queue_len``
    Peak over the episode of the total number of halting vehicles (SUMO counts a
    vehicle as halting below 0.1 m/s).
``throughput_veh``
    Number of vehicles that reached their destination within the episode.
``total_co2_kg``
    Integral of per-vehicle CO2 emission over the episode.

Sampling happens once per *simulated second*, never once per agent decision. A
sumo-rl agent acts every ``delta_time`` seconds, so sampling per decision would
undercount waiting time by a factor of ``delta_time``. Baseline controllers drive
raw TraCI, while RL controllers run inside sumo-rl; both feed this same collector
so the numbers mean the same thing.

The collector also tracks a *junction-local* scope: delay, queue and throughput on
the incoming lanes of one junction. That is what isolates the effect of a single
RL-controlled junction inside an otherwise fixed-time corridor (Week 3).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


class MetricsCollector:
    """Accumulates corridor-wide and junction-local metrics one simulated second at a time.

    Call :meth:`sample` after every ``simulationStep()``, then :meth:`corridor_row`
    and/or :meth:`junction_row` once the episode ends.
    """

    def __init__(self, conn: Any, junction_lanes: Iterable[str] | None = None) -> None:
        """Create a collector bound to a live SUMO connection.

        Args:
            conn: a TraCI connection (or the ``libsumo`` module) already connected
                to a running simulation.
            junction_lanes: incoming lane ids of the junction to track locally.
                Pass ``None`` to skip junction-local accounting.
        """
        self.conn = conn
        self.lane_ids: list[str] = list(conn.lane.getIDList())
        self.junction_lanes: list[str] = list(junction_lanes) if junction_lanes else []

        # corridor-wide state
        self._veh_stopped_s: dict[str, int] = {}
        self._completed_waits: list[int] = []
        self._max_queue = 0
        self._arrived = 0
        self._total_co2_mg = 0.0

        # junction-local state
        self._local_stopped_s: dict[str, int] = {}
        self._local_completed: list[int] = []
        self._local_max_queue = 0
        self._local_present: set[str] = set()
        self._local_throughput = 0

        self.steps_sampled = 0

    def sample(self) -> None:
        """Record one simulated second. Call immediately after ``simulationStep()``."""
        conn = self.conn
        veh_ids = conn.vehicle.getIDList()

        for vid in veh_ids:
            if conn.vehicle.getWaitingTime(vid) > 0:
                self._veh_stopped_s[vid] = self._veh_stopped_s.get(vid, 0) + 1
            self._total_co2_mg += conn.vehicle.getCO2Emission(vid)  # mg/s x 1 s

        for vid in conn.simulation.getArrivedIDList():
            self._completed_waits.append(self._veh_stopped_s.pop(vid, 0))
            self._arrived += 1

        queue = sum(conn.lane.getLastStepHaltingNumber(lid) for lid in self.lane_ids)
        if queue > self._max_queue:
            self._max_queue = queue

        if self.junction_lanes:
            self._sample_junction()

        self.steps_sampled += 1

    def _sample_junction(self) -> None:
        """Record one simulated second of junction-local state."""
        conn = self.conn
        present: set[str] = set()
        local_queue = 0
        for lane in self.junction_lanes:
            local_queue += conn.lane.getLastStepHaltingNumber(lane)
            for vid in conn.lane.getLastStepVehicleIDs(lane):
                present.add(vid)
                if conn.vehicle.getWaitingTime(vid) > 0:
                    self._local_stopped_s[vid] = self._local_stopped_s.get(vid, 0) + 1
                else:
                    self._local_stopped_s.setdefault(vid, 0)

        if local_queue > self._local_max_queue:
            self._local_max_queue = local_queue

        # A vehicle that was on an approach lane last second and is gone now has
        # cleared the junction (or left the network from that approach).
        for vid in self._local_present - present:
            self._local_completed.append(self._local_stopped_s.pop(vid, 0))
            self._local_throughput += 1
        self._local_present = present

    def corridor_row(self) -> dict[str, Any]:
        """Return the corridor-wide metrics for the finished episode.

        Returns:
            Dict with ``avg_wait_time_s``, ``max_queue_len``, ``throughput_veh``,
            ``total_co2_kg``.
        """
        avg_wait = sum(self._completed_waits) / len(self._completed_waits) if self._completed_waits else 0.0
        return {
            "avg_wait_time_s": round(avg_wait, 2),
            "max_queue_len": self._max_queue,
            "throughput_veh": self._arrived,
            "total_co2_kg": round(self._total_co2_mg / 1e6, 4),
        }

    def junction_row(self) -> dict[str, Any]:
        """Return the junction-local metrics for the finished episode.

        Returns:
            Dict with the same metric names, restricted to the tracked junction's
            incoming lanes. ``total_co2_kg`` is not meaningful per-junction and is
            reported as the corridor value's share is unknown, so it is omitted.

        Raises:
            ValueError: if the collector was built without ``junction_lanes``.
        """
        if not self.junction_lanes:
            raise ValueError("This collector was created without junction_lanes.")
        waits = self._local_completed
        avg_wait = sum(waits) / len(waits) if waits else 0.0
        return {
            "avg_wait_time_s": round(avg_wait, 2),
            "max_queue_len": self._local_max_queue,
            "throughput_veh": self._local_throughput,
        }

    def max_lane_wait_s(self) -> float:
        """Return the worst single-vehicle stopped time seen so far (fairness probe).

        Returns:
            The largest accumulated stopped-seconds value over all vehicles, whether
            they have completed their trip or not.
        """
        seen = list(self._veh_stopped_s.values()) + self._completed_waits
        return float(max(seen)) if seen else 0.0

    def wait_percentile(self, pct: float) -> float:
        """Return a percentile of completed-trip waiting times.

        Args:
            pct: percentile in ``[0, 100]``.

        Returns:
            The requested percentile, or ``0.0`` when no trip has completed.
        """
        if not self._completed_waits:
            return 0.0
        ordered = sorted(self._completed_waits)
        idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return float(ordered[idx])
