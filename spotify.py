"""Spotify integration.

Reads OAuth tokens stored in the Supabase `user_social_links` table (populated
by the Next.js front-end OAuth flow), refreshes them when expired, and queries
the Spotify Web API on behalf of the user.

All functions are best-effort: if something fails, they return a dict with an
`error` key so the chatbot can keep operating on partial data.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_ME_URL = "https://api.spotify.com/v1/me"
SPOTIFY_TOP_ARTISTS_URL = "https://api.spotify.com/v1/me/top/artists"
SPOTIFY_TOP_TRACKS_URL = "https://api.spotify.com/v1/me/top/tracks"
SPOTIFY_RECENT_URL = "https://api.spotify.com/v1/me/player/recently-played"
SPOTIFY_PLAYLISTS_URL = "https://api.spotify.com/v1/me/playlists"

# Reuse the same heuristic buckets as YouTube so interests feel consistent.
_INTEREST_KEYWORDS = {
    "música clásica": ["classical", "clásica", "sinfónica", "orchestra", "opera", "ópera"],
    "flamenco": ["flamenco", "copla", "sevillanas", "rumba"],
    "bolero / romántica": ["bolero", "balada", "romántic", "canción melódica"],
    "salsa / latino": ["salsa", "merengue", "bachata", "cumbia", "latin", "reggaeton"],
    "tango": ["tango"],
    "jazz": ["jazz", "blues", "swing"],
    "pop": ["pop"],
    "rock": ["rock", "metal", "punk"],
    "folk / cantautor": ["folk", "cantautor", "acoustic", "singer-songwriter"],
    "música religiosa": ["gospel", "religi", "christian", "liturg"],
    "electrónica": ["electronic", "house", "techno", "edm", "dance"],
    "hip hop": ["hip hop", "hip-hop", "rap", "trap"],
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
        f"?user_id=eq.{user_id}&provider=eq.spotify&select=*"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None


async def _update_link(user_id: str, patch: dict) -> None:
    url = (
        f"{_supabase_url()}/rest/v1/user_social_links"
        f"?user_id=eq.{user_id}&provider=eq.spotify"
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


def _basic_auth_header() -> Optional[str]:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    token = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {token}"


async def _refresh_access_token(user_id: str, refresh_token: str) -> Optional[str]:
    auth_header = _basic_auth_header()
    if not auth_header:
        logger.error("Spotify OAuth client not configured on backend")
        return None

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            SPOTIFY_TOKEN_URL,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
        )
    if resp.status_code != 200:
        logger.warning("Spotify token refresh failed: %s %s", resp.status_code, resp.text)
        return None

    tokens = resp.json()
    access_token = tokens.get("access_token")
    expires_in = int(tokens.get("expires_in", 3600))
    new_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()

    # Spotify sometimes issues a new refresh token; persist it if so.
    patch: dict[str, Any] = {
        "access_token": access_token,
        "expires_at": new_expires_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if tokens.get("refresh_token"):
        patch["refresh_token"] = tokens["refresh_token"]

    await _update_link(user_id, patch)
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
# API calls
# ---------------------------------------------------------------------------

def _infer_interests(texts: list[str]) -> list[str]:
    blob = " ".join(t.lower() for t in texts)
    counter: Counter[str] = Counter()
    for tag, keywords in _INTEREST_KEYWORDS.items():
        for kw in keywords:
            if kw in blob:
                counter[tag] += 1
                break
    return [tag for tag, _ in counter.most_common(6)]


async def fetch_top_artists(access_token: str, limit: int = 10) -> dict:
    params = {"limit": min(limit, 50), "time_range": "medium_term"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            SPOTIFY_TOP_ARTISTS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
    if resp.status_code != 200:
        return {"artists": [], "error": f"Spotify top artists API {resp.status_code}"}
    items = resp.json().get("items", [])
    artists = [
        {
            "id": a.get("id"),
            "name": a.get("name", ""),
            "genres": a.get("genres", []) or [],
            "popularity": a.get("popularity"),
        }
        for a in items
    ]
    return {"artists": artists}


async def fetch_top_tracks(access_token: str, limit: int = 10) -> dict:
    params = {"limit": min(limit, 50), "time_range": "medium_term"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            SPOTIFY_TOP_TRACKS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
    if resp.status_code != 200:
        return {"tracks": [], "error": f"Spotify top tracks API {resp.status_code}"}
    items = resp.json().get("items", [])
    tracks = [
        {
            "id": t.get("id"),
            "name": t.get("name", ""),
            "artists": [a.get("name", "") for a in t.get("artists", [])],
        }
        for t in items
    ]
    return {"tracks": tracks}


async def fetch_recently_played(access_token: str, limit: int = 15) -> dict:
    params = {"limit": min(limit, 50)}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            SPOTIFY_RECENT_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
    if resp.status_code != 200:
        return {"tracks": [], "error": f"Spotify recent API {resp.status_code}"}
    items = resp.json().get("items", [])
    tracks = []
    for item in items:
        t = item.get("track", {}) or {}
        tracks.append({
            "name": t.get("name", ""),
            "artists": [a.get("name", "") for a in t.get("artists", [])],
            "played_at": item.get("played_at"),
        })
    return {"tracks": tracks}


async def fetch_playlists(access_token: str, limit: int = 20) -> dict:
    params = {"limit": min(limit, 50)}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            SPOTIFY_PLAYLISTS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
    if resp.status_code != 200:
        return {"playlists": [], "error": f"Spotify playlists API {resp.status_code}"}
    items = resp.json().get("items", [])
    playlists = [
        {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "description": p.get("description") or None,
            "tracks_total": (p.get("tracks") or {}).get("total", 0),
        }
        for p in items
    ]
    return {"playlists": playlists}


# ---------------------------------------------------------------------------
# Public facade
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
        return {"connected": False, "error": "Spotify no está vinculado."}

    result: dict[str, Any] = {
        "connected": True,
        "email": link.get("provider_email"),
    }

    want_top = kind in ("all", "top")
    want_recent = kind in ("all", "recent")
    want_playlists = kind in ("all", "playlists")

    tasks = []
    if want_top:
        tasks.append(("top_artists", fetch_top_artists(access_token)))
        tasks.append(("top_tracks", fetch_top_tracks(access_token)))
    if want_recent:
        tasks.append(("recent", fetch_recently_played(access_token)))
    if want_playlists:
        tasks.append(("playlists", fetch_playlists(access_token)))

    if tasks:
        responses = await asyncio.gather(*(coro for _, coro in tasks))
        for (label, _), data in zip(tasks, responses):
            result[label] = data

    # Infer interests from top artist genres + playlist names, same as YouTube.
    interest_blobs: list[str] = []
    top_artists_data = result.get("top_artists") or {}
    for a in top_artists_data.get("artists", []):
        interest_blobs.extend(a.get("genres") or [])
        interest_blobs.append(a.get("name", ""))
    playlists_data = result.get("playlists") or {}
    for p in playlists_data.get("playlists", []):
        interest_blobs.append(p.get("name", ""))
    if interest_blobs:
        result["inferred_interests"] = _infer_interests(interest_blobs)
    else:
        result["inferred_interests"] = []

    return result


# ---------------------------------------------------------------------------
# Sync helper (used by the LangChain tool, which runs inside a running loop)
# ---------------------------------------------------------------------------

def get_user_data_sync(user_id: str, kind: str = "all") -> dict:
    """Synchronous wrapper for `get_user_data` — safe to call from inside an
    already-running event loop (LangChain tool execution).
    """
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, get_user_data(user_id, kind=kind)).result()
    except Exception as e:  # pragma: no cover — defensive
        logger.exception("spotify get_user_data_sync failed")
        return {"connected": False, "error": f"Error consultando Spotify: {e}"}


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_spotify_for_chat(data: dict) -> str:
    """Formats the Spotify payload into a short, friendly text block for the LLM."""
    if not data.get("connected"):
        reason = data.get("error") or "la cuenta de Spotify no está vinculada."
        return (
            f"No se pudo obtener la música del usuario: {reason} "
            "Sugiérele vincular Spotify desde 'Cuentas conectadas' en su perfil."
        )

    parts: list[str] = ["Actividad musical del usuario en Spotify:"]

    top_artists = (data.get("top_artists") or {}).get("artists") or []
    if top_artists:
        names = [a["name"] for a in top_artists[:6] if a.get("name")]
        if names:
            parts.append(f"- Artistas más escuchados: {', '.join(names)}.")

    top_tracks = (data.get("top_tracks") or {}).get("tracks") or []
    if top_tracks:
        snippets = []
        for t in top_tracks[:5]:
            name = t.get("name") or ""
            artists = ", ".join(t.get("artists") or [])
            if name:
                snippets.append(f"{name}" + (f" — {artists}" if artists else ""))
        if snippets:
            parts.append(f"- Canciones favoritas: {'; '.join(snippets)}.")

    recent = (data.get("recent") or {}).get("tracks") or []
    if recent:
        snippets = []
        for t in recent[:5]:
            name = t.get("name") or ""
            artists = ", ".join(t.get("artists") or [])
            if name:
                snippets.append(f"{name}" + (f" — {artists}" if artists else ""))
        if snippets:
            parts.append(f"- Escuchado recientemente: {'; '.join(snippets)}.")

    playlists = (data.get("playlists") or {}).get("playlists") or []
    if playlists:
        names = [p["name"] for p in playlists[:5] if p.get("name")]
        if names:
            parts.append(f"- Playlists: {', '.join(names)}.")

    interests = data.get("inferred_interests") or []
    if interests:
        parts.append(f"- Géneros/intereses inferidos: {', '.join(interests)}.")

    if len(parts) == 1:
        parts.append("- No hay datos musicales suficientes todavía.")

    return "\n".join(parts)
