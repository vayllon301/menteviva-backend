import asyncio
import logging
import os
from datetime import datetime, timezone

from croniter import croniter
from twilio.rest import Client as TwilioClient

from reminders import (
    create_notification,
    get_due_reminders,
    update_reminder,
)

logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_SMS_FROM = os.getenv("TWILIO_SMS_FROM", "")

POLL_INTERVAL_SECONDS = 30


def _send_sms(to: str, body: str) -> bool:
    """Send an SMS via Twilio. Returns True on success."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_SMS_FROM:
        logger.warning("Twilio SMS credentials not configured, skipping SMS")
        return False
    if not to:
        logger.warning("No recipient phone number, skipping SMS")
        return False
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=body, from_=TWILIO_SMS_FROM, to=to)
        return True
    except Exception:
        logger.exception("Failed to send SMS to %s", to)
        return False


def _next_occurrence(cron_expr: str, after: datetime) -> str:
    """Compute the next occurrence from a cron expression after a given datetime."""
    cron = croniter(cron_expr, after)
    next_dt = cron.get_next(datetime)
    return next_dt.isoformat()


async def _get_user_phone(user_id: str) -> str:
    """Fetch the user's phone number from user_profile in Supabase."""
    import httpx
    from reminders import _supabase_headers, _supabase_url

    url = (
        f"{_supabase_url()}/rest/v1/user_profile"
        f"?id=eq.{user_id}&select=number"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        rows = resp.json()
        if rows and rows[0].get("number"):
            return rows[0]["number"]
    return ""


async def _process_due_reminder(reminder: dict) -> None:
    """Handle a single due reminder: send SMS, create notification, update status."""
    reminder_id = reminder["id"]
    user_id = reminder["user_id"]
    message = reminder["message"]
    recurrence = reminder.get("recurrence")

    # 1. Send SMS
    phone = await _get_user_phone(user_id)
    sms_body = f"MenteViva - Recordatorio: {message}"
    _send_sms(phone, sms_body)

    # 2. Create in-app notification
    await create_notification(user_id, reminder_id, message)

    # 3. Update reminder status
    if recurrence:
        now = datetime.now(timezone.utc)
        next_at = _next_occurrence(recurrence, now)
        await update_reminder(reminder_id, {"remind_at": next_at})
        logger.info("Recurring reminder %s advanced to %s", reminder_id, next_at)
    else:
        await update_reminder(reminder_id, {"status": "completed"})
        logger.info("One-time reminder %s completed", reminder_id)


async def run_tick() -> int:
    """Run one polling iteration. Returns the number of reminders processed."""
    processed = 0
    try:
        due = await get_due_reminders()
        if due:
            logger.info("Found %d due reminder(s)", len(due))
        for reminder in due:
            try:
                await _process_due_reminder(reminder)
                processed += 1
            except Exception:
                logger.exception(
                    "Error processing reminder %s", reminder.get("id")
                )
    except Exception:
        logger.exception("Error in scheduler tick")
    return processed


async def scheduler_loop() -> None:
    """In-process scheduler loop. Used in local dev; in production an external
    cron should hit /scheduler/tick instead (see RUN_INPROCESS_SCHEDULER)."""
    logger.info("Reminder scheduler started (poll every %ds)", POLL_INTERVAL_SECONDS)
    while True:
        await run_tick()
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
