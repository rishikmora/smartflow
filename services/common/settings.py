"""Shared settings for the SmartFlow services.

Every service reads its configuration from environment variables through this
module rather than from scattered ``os.environ`` calls, so a deployment can be
described entirely by its environment.

Data is **mounted, not baked**. The services read the project's committed
metrics, graph and detector results from ``SMARTFLOW_DATA_ROOT``, which Docker
Compose and the Kubernetes manifests both point at a read-only mount of the
repository. Baking a copy of ``outputs/`` into five images would multiply a
hundred megabytes five ways and go stale the moment a benchmark is re-run.
"""

from __future__ import annotations

import os


def _root() -> str:
    """Locate the mounted project data.

    Returns:
        Absolute path to the directory holding ``outputs/`` and ``data/``.
    """
    explicit = os.environ.get("SMARTFLOW_DATA_ROOT")
    if explicit:
        return explicit
    # Running outside a container, from the repository itself.
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return here


DATA_ROOT = _root()
OUTPUTS_DIR = os.path.join(DATA_ROOT, "outputs")
DATA_DIR = os.path.join(DATA_ROOT, "data")

SERVICE_PORT = int(os.environ.get("PORT", "8000"))

# ── authentication ──────────────────────────────────────────────────────────
# Auth0 is used when a tenant is configured. Without one the services fall back
# to a local HS256 dev token so the authenticated path is still exercised and
# testable; that fallback is refused when SMARTFLOW_ENV=production.
AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN", "")
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE", "")
DEV_JWT_SECRET = os.environ.get("SMARTFLOW_DEV_JWT_SECRET", "smartflow-dev-secret")
ENVIRONMENT = os.environ.get("SMARTFLOW_ENV", "development")
AUTH_DISABLED = os.environ.get("SMARTFLOW_AUTH_DISABLED", "").lower() in {"1", "true", "yes"}

# ── events ──────────────────────────────────────────────────────────────────
# Redpanda speaks the Kafka API. When no broker is reachable the publisher
# degrades to a no-op rather than taking the service down with it: an analytics
# event that cannot be delivered is not a reason to fail a read request.
REDPANDA_BROKERS = os.environ.get("REDPANDA_BROKERS", "")
EVENT_TOPIC = os.environ.get("SMARTFLOW_EVENT_TOPIC", "smartflow.events")


def auth0_configured() -> bool:
    """Report whether a real Auth0 tenant is configured.

    Returns:
        True when both the domain and audience are set.
    """
    return bool(AUTH0_DOMAIN and AUTH0_AUDIENCE)
