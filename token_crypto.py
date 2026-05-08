"""AES-256-GCM helpers for stored OAuth tokens.

The same key (base64-encoded 32 bytes) is used by the Next.js callback in
`menteviva_front/lib/crypto/tokens.ts`. Layout: 12-byte nonce || 16-byte tag ||
ciphertext, base64-encoded.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_LEN = 12
TAG_LEN = 16


@lru_cache(maxsize=1)
def _key() -> bytes:
    raw = os.getenv("OAUTH_TOKEN_ENC_KEY", "")
    if not raw:
        raise RuntimeError("OAUTH_TOKEN_ENC_KEY is not configured")
    key = base64.b64decode(raw)
    if len(key) != 32:
        raise RuntimeError(
            "OAUTH_TOKEN_ENC_KEY must decode to 32 bytes (base64 of a 256-bit key)"
        )
    return key


def encrypt_token(plaintext: str) -> str:
    nonce = os.urandom(NONCE_LEN)
    aes = AESGCM(_key())
    ciphertext = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
    # cryptography appends the auth tag to the ciphertext; split it so the
    # on-disk layout matches the front-end (nonce || tag || ct).
    body, tag = ciphertext[:-TAG_LEN], ciphertext[-TAG_LEN:]
    return base64.b64encode(nonce + tag + body).decode("ascii")


def decrypt_token(blob: Optional[str]) -> Optional[str]:
    if not blob:
        return None
    raw = base64.b64decode(blob)
    if len(raw) < NONCE_LEN + TAG_LEN:
        raise ValueError("ciphertext too short")
    nonce = raw[:NONCE_LEN]
    tag = raw[NONCE_LEN : NONCE_LEN + TAG_LEN]
    body = raw[NONCE_LEN + TAG_LEN :]
    aes = AESGCM(_key())
    plaintext = aes.decrypt(nonce, body + tag, None)
    return plaintext.decode("utf-8")
