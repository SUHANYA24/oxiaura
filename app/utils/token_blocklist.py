"""JWT revocation blocklist.

Tokens are revoked on logout by storing their ``jti`` (unique token id) here.
The JWT ``token_in_blocklist_loader`` (registered in the app factory) consults
this store on every protected request.

Phase 3 uses a process-local in-memory store, which is sufficient for a single
worker in development. Phase 7 introduces Redis; this module is intentionally a
thin, swappable abstraction so the backing store can move to Redis (with native
TTL) without touching callers.
"""

import time
import threading

# Maps jti -> unix expiry timestamp (or None if unknown). Guarded by a lock so
# concurrent requests under a threaded dev server don't corrupt it.
_blocklist: dict[str, float | None] = {}
_lock = threading.Lock()


def add_to_blocklist(jti: str, expires_at: float | None = None) -> None:
    """Revoke a token by its ``jti``. ``expires_at`` is the JWT ``exp`` claim."""
    with _lock:
        _blocklist[jti] = expires_at


def is_blocklisted(jti: str) -> bool:
    """Return True if ``jti`` has been revoked and has not naturally expired."""
    with _lock:
        _prune_locked()
        return jti in _blocklist


def _prune_locked() -> None:
    """Drop entries whose token has already expired (caller holds the lock).

    Once a token is past its own ``exp`` it is rejected as expired anyway, so we
    needn't keep tracking it — this bounds memory growth.
    """
    now = time.time()
    expired = [jti for jti, exp in _blocklist.items() if exp is not None and exp < now]
    for jti in expired:
        _blocklist.pop(jti, None)


def clear() -> None:
    """Empty the blocklist. Intended for test isolation."""
    with _lock:
        _blocklist.clear()
