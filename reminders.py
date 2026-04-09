import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import quote

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


async def create_reminder(
    user_id: str,
    message: str,
    remind_at: str,
    recurrence: Optional[str] = None,
    created_by: str = "user",
) -> dict:
    """Insert a new reminder row. Returns the created row."""
    url = f"{_supabase_url()}/rest/v1/reminders"
    headers = {**_supabase_headers(), "Prefer": "return=representation"}
    payload = {
        "user_id": user_id,
        "message": message,
        "remind_at": remind_at,
        "recurrence": recurrence,
        "created_by": created_by,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else payload


async def list_active_reminders(user_id: str) -> List[dict]:
    """Return all active reminders for a user, ordered by remind_at."""
    url = (
        f"{_supabase_url()}/rest/v1/reminders"
        f"?user_id=eq.{user_id}"
        f"&status=in.(active,snoozed)"
        f"&order=remind_at.asc"
        f"&select=id,message,remind_at,recurrence,created_by,status"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        return resp.json()


async def get_due_reminders() -> List[dict]:
    """Return all reminders whose remind_at is in the past and status is active."""
    now_iso = quote(datetime.now(timezone.utc).isoformat(), safe="")
    url = (
        f"{_supabase_url()}/rest/v1/reminders"
        f"?remind_at=lte.{now_iso}"
        f"&status=eq.active"
        f"&select=*"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        return resp.json()


async def update_reminder(reminder_id: str, updates: dict) -> None:
    """Patch a reminder row by ID."""
    url = f"{_supabase_url()}/rest/v1/reminders?id=eq.{reminder_id}"
    payload = {**updates}
    async with httpx.AsyncClient() as client:
        resp = await client.patch(url, headers=_supabase_headers(), json=payload)
        resp.raise_for_status()


async def create_notification(user_id: str, reminder_id: str, message: str) -> None:
    """Insert a notification row for the in-app popup."""
    url = f"{_supabase_url()}/rest/v1/notifications"
    payload = {
        "user_id": user_id,
        "reminder_id": reminder_id,
        "message": message,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_supabase_headers(), json=payload)
        resp.raise_for_status()


async def get_unread_notifications(user_id: str) -> List[dict]:
    """Return unread notifications for a user."""
    url = (
        f"{_supabase_url()}/rest/v1/notifications"
        f"?user_id=eq.{user_id}"
        f"&is_read=eq.false"
        f"&order=created_at.desc"
        f"&select=id,reminder_id,message,created_at"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        return resp.json()


async def mark_notification_read(notification_id: str) -> None:
    """Mark a notification as read."""
    url = f"{_supabase_url()}/rest/v1/notifications?id=eq.{notification_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            url, headers=_supabase_headers(), json={"is_read": True}
        )
        resp.raise_for_status()
