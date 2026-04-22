"""Google / YouTube integration.

Reads OAuth tokens stored in the Supabase `user_social_links` table (populated
by the Next.js front-end OAuth flow), refreshes them when expired, and queries
Google Calendar + YouTube on behalf of the user.

All functions are best-effort: if something fails, they return a dict with an
`error` key so the chatbot can keep operating on partial data.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
CALENDAR_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
YOUTUBE_SUBSCRIPTIONS_URL = "https://www.googleapis.com/youtube/v3/subscriptions"

# Heuristic keyword buckets to turn raw subscription titles into interest tags
# without spending an LLM call on every request.
_INTEREST_KEYWORDS = {
    "música": ["music", "música", "canciones", "songs", "flamenco", "salsa", "tango", "copla", "bolero"],
    "cocina": ["cocina", "recetas", "cooking", "food", "chef", "bake"],
    "jardinería": ["jardín", "jardineria", "garden", "plantas", "huerto"],
    "manualidades": ["diy", "manualidades", "ganchillo", "crochet", "punto", "knitting"],
    "noticias": ["news", "noticias", "telediario", "informativos"],
    "religión": ["iglesia", "catolic", "misa", "rosario", "biblia"],
    "salud": ["salud", "health", "medic", "doctor", "wellness"],
    "viajes": ["viaje", "travel", "turismo", "vuelta al mundo"],
    "historia": ["history", "historia", "documental", "documentary"],
    "cine": ["cine", "movie", "film", "pelicul"],
    "deportes": ["deporte", "sport", "fútbol", "futbol", "tenis"],
    "tecnología": ["tech", "tecnolog", "informática", "computer"],
}


# ---------------------------------------------------------------------------
# Supabase helpers (service-role, bypasses RLS)
# ---------------------------------------------------------------------------

def _supabase_headers() -> dict:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supabase_url() -> str:
    return os.getenv("SUPABASE_URL", "")


async def _load_link(user_id: str) -> Optional[dict]:
    url = (
        f"{_supabase_url()}/rest/v1/user_social_links"
        f"?user_id=eq.{user_id}&provider=eq.google&select=*"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None


async def _update_link(user_id: str, patch: dict) -> None:
    url = (
        f"{_supabase_url()}/rest/v1/user_social_links"
        f"?user_id=eq.{user_id}&provider=eq.google"
    )
    headers = {**_supabase_headers(), "Prefer": "return=minimal"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(url, headers=headers, json=patch)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

def _is_expired(expires_at: Optional[str]) -> bool:
    if not expires_at:
        return True
    try:
        dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    # Refresh with a 60 s safety margin.
    return datetime.now(timezone.utc) >= dt - timedelta(seconds=60)


async def _refresh_access_token(user_id: str, refresh_token: str) -> Optional[str]:
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.error("Google OAuth client not configured on backend")
        return None

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if resp.status_code != 200:
        logger.warning("Google token refresh failed: %s %s", resp.status_code, resp.text)
        return None

    tokens = resp.json()
    access_token = tokens.get("access_token")
    expires_in = int(tokens.get("expires_in", 3600))
    new_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()

    await _update_link(user_id, {
        "access_token": access_token,
        "expires_at": new_expires_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return access_token


async def _get_valid_access_token(user_id: str) -> tuple[Optional[str], Optional[dict]]:
    """Returns (access_token, link_row) or (None, None) if not connected."""
    link = await _load_link(user_id)
    if not link:
        return None, None

    if _is_expired(link.get("expires_at")):
        refresh_token = link.get("refresh_token")
        if not refresh_token:
            return None, link
        new_access = await _refresh_access_token(user_id, refresh_token)
        if not new_access:
            return None, link
        link["access_token"] = new_access

    return link["access_token"], link


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

async def fetch_upcoming_events(access_token: str, max_results: int = 10) -> dict:
    time_min = datetime.now(timezone.utc).isoformat()
    params = {
        "timeMin": time_min,
        "maxResults": max_results,
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            CALENDAR_EVENTS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
    if resp.status_code != 200:
        return {"events": [], "error": f"Calendar API {resp.status_code}"}

    items = resp.json().get("items", [])
    events = []
    for item in items:
        start = item.get("start", {})
        end = item.get("end", {})
        events.append({
            "id": item.get("id"),
            "summary": item.get("summary", ""),
            "start": start.get("dateTime") or start.get("date"),
            "end": end.get("dateTime") or end.get("date"),
            "location": item.get("location"),
        })
    return {"events": events}


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def _infer_interests(titles: list[str]) -> list[str]:
    blob = " ".join(t.lower() for t in titles)
    counter: Counter[str] = Counter()
    for tag, keywords in _INTEREST_KEYWORDS.items():
        for kw in keywords:
            if kw in blob:
                counter[tag] += 1
                break
    return [tag for tag, _ in counter.most_common(6)]


async def fetch_youtube_subscriptions(access_token: str, max_results: int = 25) -> dict:
    params = {
        "part": "snippet",
        "mine": "true",
        "maxResults": min(max_results, 50),
        "order": "relevance",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            YOUTUBE_SUBSCRIPTIONS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
    if resp.status_code != 200:
        return {"subscriptions": [], "inferred_interests": [], "error": f"YouTube API {resp.status_code}"}

    items = resp.json().get("items", [])
    subs = []
    for item in items:
        snippet = item.get("snippet", {})
        resource = snippet.get("resourceId", {})
        subs.append({
            "channel_id": resource.get("channelId", ""),
            "title": snippet.get("title", ""),
            "description": snippet.get("description") or None,
        })

    interests = _infer_interests([s["title"] for s in subs])
    return {"subscriptions": subs, "inferred_interests": interests}


# ---------------------------------------------------------------------------
# Public facade used by main.py
# ---------------------------------------------------------------------------

async def get_status(user_id: str) -> dict:
    link = await _load_link(user_id)
    if not link:
        return {"connected": False}
    return {
        "connected": True,
        "email": link.get("provider_email"),
        "scopes": link.get("scopes", []),
        "connected_at": link.get("connected_at"),
    }


async def get_user_data(user_id: str, kind: str = "all") -> dict:
    access_token, link = await _get_valid_access_token(user_id)
    if not access_token or not link:
        return {"connected": False, "error": "Google no está vinculado."}

    scopes: list[str] = link.get("scopes", [])
    has_calendar = any(s.endswith("/auth/calendar.readonly") for s in scopes)
    has_youtube = any(s.endswith("/auth/youtube.readonly") for s in scopes)

    result: dict[str, Any] = {
        "connected": True,
        "email": link.get("provider_email"),
    }

    want_calendar = kind in ("all", "calendar") and has_calendar
    want_youtube = kind in ("all", "youtube") and has_youtube

    if want_calendar:
        result["calendar"] = await fetch_upcoming_events(access_token)
    if want_youtube:
        result["youtube"] = await fetch_youtube_subscriptions(access_token)

    return result
