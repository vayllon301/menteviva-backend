"""Per-user fixed-window rate limiter.

In-process — sufficient for single-worker deployments. Swap to Redis before
horizontal scaling. Used by routes that hit paid APIs (Twilio, OpenAI, xAI).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    remaining: int
    reset_at: float


_buckets: dict[str, _Bucket] = {}
_lock = threading.Lock()


def check(key: str, limit: int, window_seconds: float) -> tuple[bool, int, float]:
    """Returns (allowed, remaining, reset_at_epoch)."""
    now = time.time()
    with _lock:
        bucket = _buckets.get(key)
        if bucket is None or bucket.reset_at <= now:
            bucket = _Bucket(remaining=limit - 1, reset_at=now + window_seconds)
            _buckets[key] = bucket
            return True, bucket.remaining, bucket.reset_at
        if bucket.remaining <= 0:
            return False, 0, bucket.reset_at
        bucket.remaining -= 1
        return True, bucket.remaining, bucket.reset_at


def _gc_locked(now: float) -> None:
    expired = [k for k, b in _buckets.items() if b.reset_at <= now]
    for k in expired:
        _buckets.pop(k, None)
