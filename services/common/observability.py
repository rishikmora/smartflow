"""Prometheus instrumentation shared by the SmartFlow services.

One call per service exposes ``/metrics`` with request counts, latencies and
status codes, which is what ``deploy/prometheus/prometheus.yml`` scrapes and what
the Grafana dashboard renders.

If the instrumentator is not installed the service still starts and serves its
domain endpoints. Monitoring is valuable, but a missing metrics library is not a
reason for a corridor API to be down.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def instrument(app: Any, service: str) -> bool:
    """Expose Prometheus metrics on ``/metrics``.

    Args:
        app: the FastAPI application.
        service: the service name, attached as a label.

    Returns:
        True if instrumentation was installed, False if it was unavailable.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        log.warning("prometheus_fastapi_instrumentator missing; /metrics disabled.")
        return False
    try:
        Instrumentator(
            should_group_status_codes=False,
            excluded_handlers=["/metrics", "/health"],
        ).instrument(app).expose(app, include_in_schema=False)
        log.info("Prometheus metrics exposed for %s", service)
        return True
    except Exception as exc:  # noqa: BLE001 - never fail startup over telemetry
        log.warning("Could not install metrics for %s: %s", service, exc)
        return False
