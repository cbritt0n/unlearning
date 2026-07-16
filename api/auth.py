"""
API authentication and production secret guards.

Environment
-----------
HEALER_API_KEY
    If set, mutating and enterprise endpoints require
    ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
HEALER_ENV
    ``production`` (or ``prod``) enables strict checks:
    - default signing key is rejected
    - API key is required (unless HEALER_ALLOW_INSECURE=1)
HEALER_REQUIRE_AUTH
    ``1`` / ``true`` forces API key requirement even in dev.
HEALER_ALLOW_INSECURE
    ``1`` bypasses production auth/signing guards (tests only).
HEALER_SIGNING_KEY
    HMAC key for receipts; must not be the default in production.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

DEFAULT_SIGNING_KEY = "dev-only-placeholder-signing-key"

# Paths that remain reachable without an API key (liveness / docs).
PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
    }
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_production() -> bool:
    env = os.environ.get("HEALER_ENV", "development").strip().lower()
    return env in ("production", "prod")


def allow_insecure() -> bool:
    return env_flag("HEALER_ALLOW_INSECURE", False)


def get_api_key() -> str | None:
    key = os.environ.get("HEALER_API_KEY", "").strip()
    return key or None


def auth_required() -> bool:
    if allow_insecure():
        return False
    if env_flag("HEALER_REQUIRE_AUTH", False):
        return True
    if get_api_key() is not None:
        return True
    if is_production():
        return True
    return False


def get_signing_key_bytes() -> bytes:
    raw = os.environ.get("HEALER_SIGNING_KEY", DEFAULT_SIGNING_KEY)
    return raw.encode("utf-8") if not isinstance(raw, bytes) else raw


def validate_production_secrets() -> None:
    """
    Raise RuntimeError if production configuration is unsafe.

    Called at API lifespan startup.
    """
    if allow_insecure():
        logger.warning("HEALER_ALLOW_INSECURE=1 — secret guards disabled")
        return

    signing = os.environ.get("HEALER_SIGNING_KEY", DEFAULT_SIGNING_KEY)
    if is_production() and signing == DEFAULT_SIGNING_KEY:
        raise RuntimeError(
            "HEALER_ENV=production requires a non-default HEALER_SIGNING_KEY. "
            "Set a long random secret before starting the API."
        )

    if is_production() and get_api_key() is None:
        raise RuntimeError(
            "HEALER_ENV=production requires HEALER_API_KEY "
            "(or HEALER_ALLOW_INSECURE=1 for emergency bypass)."
        )

    if env_flag("HEALER_REQUIRE_STRONG_SIGNING", False) and (
        signing == DEFAULT_SIGNING_KEY
    ):
        raise RuntimeError(
            "HEALER_REQUIRE_STRONG_SIGNING=1 rejects the default signing key"
        )


def extract_presented_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def keys_match(presented: str, expected: str) -> bool:
    """Constant-time compare when lengths match."""
    try:
        return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return False


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Enforce API key on non-public routes when auth is required.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path.rstrip("/") or "/"
        # Normalize: allow /health/ etc.
        if path in PUBLIC_PATHS or path.startswith("/docs"):
            return await call_next(request)

        if not auth_required():
            return await call_next(request)

        expected = get_api_key()
        if expected is None:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "auth required but HEALER_API_KEY is not configured"
                },
            )

        presented = extract_presented_key(request)
        if not presented or not keys_match(presented, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def assert_api_key_dependency(request: Request) -> None:
    """Optional FastAPI dependency alternative to middleware."""
    if not auth_required():
        return
    expected = get_api_key()
    if expected is None:
        raise HTTPException(
            status_code=503,
            detail="auth required but HEALER_API_KEY is not configured",
        )
    presented = extract_presented_key(request)
    if not presented or not keys_match(presented, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
