"""HMAC-signed QR tokens for agreement authenticity (BUILD_SPEC Phase 8, §3.8).

A token binds an agreement number to a signature the server can verify without
trusting the client. Anyone can *read* a token (it is printed as a QR on the
PDF), but only the holder of ``QR_SECRET_KEY`` can *mint* one, so a tampered or
forged token fails verification at the public ``/verify/{token}`` endpoint.

Token format:  ``<base64url(payload)>.<hex hmac-sha256(payload)>``
where ``payload`` is ``"<agreement_number>:<random nonce>"``. The nonce makes
tokens unguessable and unique even for identical inputs. No HTTP types here.
"""

from __future__ import annotations

import base64
import hmac
import secrets
from hashlib import sha256

from flask import current_app


def _secret() -> bytes:
    return current_app.config["QR_SECRET_KEY"].encode("utf-8")


def _sign(payload: bytes) -> str:
    return hmac.new(_secret(), payload, sha256).hexdigest()


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def make_token(agreement_number: str) -> str:
    """Mint a signed token for ``agreement_number`` (with a random nonce)."""
    payload = f"{agreement_number}:{secrets.token_hex(8)}".encode("utf-8")
    return f"{_b64encode(payload)}.{_sign(payload)}"


def verify_token(token: str) -> str | None:
    """Return the agreement number if ``token`` is authentic, else ``None``.

    Uses a constant-time comparison so a forged signature cannot be discovered
    by timing. Any malformed input returns ``None`` rather than raising.
    """
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    try:
        payload = _b64decode(encoded)
    except (ValueError, base64.binascii.Error):
        return None

    expected = _sign(payload)
    if not hmac.compare_digest(expected, signature):
        return None

    agreement_number, _, _nonce = payload.decode("utf-8", "replace").partition(":")
    return agreement_number or None
