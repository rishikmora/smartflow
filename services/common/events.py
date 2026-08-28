"""Event publishing to Redpanda, shared by every SmartFlow service.

Redpanda speaks the Kafka API, so a standard Kafka client talks to it unchanged.
The publisher is deliberately **best-effort**: if no broker is configured or the
broker is unreachable, publishing becomes a no-op and the caller carries on.

That is a decision, not an oversight. These events are analytics — "a query was
answered", "an anomaly was flagged" — and a read request should not fail because
a message bus is down. Anything that genuinely must not be lost does not belong
on this path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from . import settings

log = logging.getLogger(__name__)

_producer: Any = None
_producer_lock = threading.Lock()
_unavailable_logged = False


def _get_producer() -> Any:
    """Return a Kafka producer, creating it on first use.

    Returns:
        A ``KafkaProducer``, or None when unavailable.
    """
    global _producer, _unavailable_logged

    if not settings.REDPANDA_BROKERS:
        return None
    if _producer is not None:
        return _producer

    with _producer_lock:
        if _producer is not None:
            return _producer
        try:
            from kafka import KafkaProducer

            _producer = KafkaProducer(
                bootstrap_servers=settings.REDPANDA_BROKERS.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=1,
                request_timeout_ms=3000,
                max_block_ms=3000,
            )
            log.info("Connected to Redpanda at %s", settings.REDPANDA_BROKERS)
        except Exception as exc:  # noqa: BLE001 - broker may simply be absent
            if not _unavailable_logged:
                log.warning("Redpanda unavailable (%s); events will be dropped.", exc)
                _unavailable_logged = True
            _producer = None
    return _producer


def publish(service: str, event: str, payload: dict[str, Any] | None = None) -> bool:
    """Publish one analytics event.

    Args:
        service: the emitting service's name.
        event: a short event name, e.g. ``"query.answered"``.
        payload: optional JSON-serialisable detail.

    Returns:
        True if the event was handed to the broker, False if it was dropped.
    """
    producer = _get_producer()
    if producer is None:
        return False
    message = {
        "service": service,
        "event": event,
        "ts": time.time(),
        "payload": payload or {},
    }
    try:
        producer.send(settings.EVENT_TOPIC, message)
        return True
    except Exception as exc:  # noqa: BLE001 - never fail a request over telemetry
        log.warning("Dropping event %s/%s: %s", service, event, exc)
        return False


def broker_status() -> dict[str, Any]:
    """Describe the current event-publishing state.

    Returns:
        A small status dictionary suitable for a health endpoint.
    """
    # "producer_initialised" rather than "connected": the producer is created
    # lazily on first publish, so a freshly started service reports False here
    # even though the broker is perfectly reachable. Calling that "connected:
    # false" on a health endpoint reads as an outage that is not happening.
    return {
        "configured": bool(settings.REDPANDA_BROKERS),
        "brokers": settings.REDPANDA_BROKERS or None,
        "topic": settings.EVENT_TOPIC,
        "producer_initialised": _producer is not None,
    }


def close() -> None:
    """Flush and close the producer, if one was created.

    Called from each service's shutdown hook so buffered events are delivered
    instead of raising a timeout during interpreter teardown.
    """
    global _producer
    if _producer is None:
        return
    try:
        _producer.flush(timeout=3)
        _producer.close(timeout=3)
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        log.warning("Error closing the event producer: %s", exc)
    finally:
        _producer = None
