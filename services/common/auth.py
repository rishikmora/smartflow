"""Bearer-token authentication shared by every SmartFlow service.

Two modes, chosen by configuration rather than by a code branch a deployment can
forget to flip:

* **Auth0** — when ``AUTH0_DOMAIN`` and ``AUTH0_AUDIENCE`` are set, tokens are
  verified as RS256 against the tenant's published JWKS, checking signature,
  audience and issuer. This is the path a real deployment uses.
* **Local development** — without a tenant, an HS256 token signed with a local
  secret is accepted so the authenticated path is still exercised end to end and
  can be tested in CI without provisioning an identity provider.

The dev fallback is **refused outright when ``SMARTFLOW_ENV=production``**. A
convenience that silently survives into production is how an unauthenticated
dashboard happens, so it fails closed instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import settings

log = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)
_jwks_cache: dict[str, Any] = {"fetched_at": 0.0, "keys": None}
JWKS_TTL_S = 3600.0


def _fetch_jwks() -> dict[str, Any]:
    """Fetch and cache the Auth0 tenant's JSON Web Key Set.

    Returns:
        The parsed JWKS document.

    Raises:
        HTTPException: if the JWKS cannot be retrieved.
    """
    now = time.time()
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"]) < JWKS_TTL_S:
        return _jwks_cache["keys"]

    import urllib.request
    import json

    url = f"https://{settings.AUTH0_DOMAIN}/.well-known/jwks.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            keys = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - network boundary
        log.error("Could not fetch JWKS from %s: %s", url, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity provider unreachable",
        ) from exc

    _jwks_cache.update({"fetched_at": now, "keys": keys})
    return keys


def _verify_auth0(token: str) -> dict[str, Any]:
    """Verify an RS256 token against the configured Auth0 tenant.

    Args:
        token: the raw bearer token.

    Returns:
        The verified claims.

    Raises:
        HTTPException: if the token is invalid, or PyJWT is unavailable.
    """
    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT is not installed in this image",
        ) from exc

    _fetch_jwks()
    try:
        client = PyJWKClient(f"https://{settings.AUTH0_DOMAIN}/.well-known/jwks.json")
        signing_key = client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.AUTH0_AUDIENCE,
            issuer=f"https://{settings.AUTH0_DOMAIN}/",
        )
    except Exception as exc:  # noqa: BLE001 - any verification failure is a 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _verify_dev(token: str) -> dict[str, Any]:
    """Verify a locally issued HS256 development token.

    Args:
        token: the raw bearer token.

    Returns:
        The verified claims.

    Raises:
        HTTPException: in production, or if the token is invalid.
    """
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No identity provider configured and dev tokens are refused "
                   "in production. Set AUTH0_DOMAIN and AUTH0_AUDIENCE.",
        )
    try:
        import jwt
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT is not installed in this image",
        ) from exc

    try:
        return jwt.decode(token, settings.DEV_JWT_SECRET, algorithms=["HS256"],
                          audience="smartflow-dev")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid development token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def issue_dev_token(subject: str = "viva-demo", ttl_s: int = 8 * 3600) -> str:
    """Mint a development token for local use and tests.

    Args:
        subject: the ``sub`` claim.
        ttl_s: lifetime in seconds.

    Returns:
        An encoded HS256 token.

    Raises:
        RuntimeError: if called with ``SMARTFLOW_ENV=production``.
    """
    if settings.ENVIRONMENT == "production":
        raise RuntimeError("Refusing to mint a development token in production.")
    import jwt

    now = int(time.time())
    return jwt.encode(
        {"sub": subject, "aud": "smartflow-dev", "iat": now, "exp": now + ttl_s,
         "scope": "read:metrics read:graph read:vision"},
        settings.DEV_JWT_SECRET,
        algorithm="HS256",
    )


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """FastAPI dependency enforcing a valid bearer token.

    Args:
        credentials: parsed Authorization header, injected by FastAPI.

    Returns:
        The verified claims, or an anonymous principal when auth is disabled.

    Raises:
        HTTPException: if the token is missing or invalid.
    """
    if settings.AUTH_DISABLED:
        if settings.ENVIRONMENT == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SMARTFLOW_AUTH_DISABLED is not permitted in production.",
            )
        return {"sub": "anonymous", "auth": "disabled"}

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    claims = (_verify_auth0(token) if settings.auth0_configured()
              else _verify_dev(token))
    return claims


def auth_mode() -> str:
    """Describe which authentication mode is active.

    Returns:
        ``"auth0"``, ``"development"`` or ``"disabled"``.
    """
    if settings.AUTH_DISABLED:
        return "disabled"
    return "auth0" if settings.auth0_configured() else "development"
