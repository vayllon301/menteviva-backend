"""Authentication helpers.

`get_current_user_id` validates the Supabase access-token JWT supplied by the
front-end (or any direct caller) against the Supabase Auth API and returns the
user UUID. All routes that touch user-specific data must depend on it.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


_USER_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 60.0  # seconds — short cache so revoked sessions clear quickly


def _supabase_url() -> str:
    return os.getenv("SUPABASE_URL", "")


def _anon_key() -> str:
    return os.getenv("SUPABASE_ANON_KEY", "")


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def _resolve_user_id(jwt: str) -> Optional[str]:
    cached = _USER_CACHE.get(jwt)
    now = time.monotonic()
    if cached and cached[1] > now:
        return cached[0]

    base = _supabase_url()
    anon = _anon_key()
    if not base or not anon:
        logger.error("Supabase URL/anon key missing — cannot validate JWT")
        return None

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{base}/auth/v1/user",
                headers={
                    "apikey": anon,
                    "Authorization": f"Bearer {jwt}",
                },
            )
    except httpx.HTTPError as e:
        logger.warning("Supabase auth lookup failed: %s", e)
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    user_id = data.get("id")
    if not user_id:
        return None

    _USER_CACHE[jwt] = (user_id, now + _CACHE_TTL)
    if len(_USER_CACHE) > 5_000:
        # Drop expired entries; cheap one-shot cleanup.
        for k, (_, exp) in list(_USER_CACHE.items()):
            if exp <= now:
                _USER_CACHE.pop(k, None)
    return user_id


async def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency: returns the authenticated user UUID or raises 401."""
    jwt = _extract_bearer(authorization)
    if not jwt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta token de autenticación",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = await _resolve_user_id(jwt)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


async def resolve_user_id_from_jwt(jwt: Optional[str]) -> Optional[str]:
    """Same logic without raising — for the Realtime websocket init flow."""
    if not jwt:
        return None
    return await _resolve_user_id(jwt)
