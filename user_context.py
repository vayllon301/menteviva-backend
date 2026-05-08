"""Server-side fetch of user profile, tutor profile, and memory by user_id.

Routes must NOT trust the frontend to ship these blobs — derive them from the
authenticated user_id using the service role key (RLS bypassed for the backend).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _supabase_headers() -> dict:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supabase_url() -> str:
    return os.getenv("SUPABASE_URL", "")


async def _select_one(table: str, user_id: str, columns: str = "*") -> Optional[dict]:
    base = _supabase_url()
    if not base:
        return None
    url = f"{base}/rest/v1/{table}?id=eq.{user_id}&select={columns}"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=_supabase_headers())
            resp.raise_for_status()
            rows = resp.json()
            return rows[0] if rows else None
    except httpx.HTTPError as e:
        logger.warning("Supabase fetch %s failed: %s", table, e)
        return None


async def fetch_user_profile(user_id: str) -> Optional[dict]:
    return await _select_one(
        "user_profile",
        user_id,
        columns="id,name,number,description,interests,city",
    )


async def fetch_tutor_profile(user_id: str) -> Optional[dict]:
    return await _select_one(
        "tutor_profile",
        user_id,
        columns="id,name,number,description,instagram,facebook,relationship,factors",
    )


async def fetch_user_memory(user_id: str) -> Optional[dict]:
    return await _select_one(
        "user_memory",
        user_id,
        columns="id,narrative,facts,updated_at",
    )
