"""Security helpers: password hashing and role-based access control.

Per BUILD_SPEC section 3:
  * Passwords are hashed with bcrypt at cost factor 12. Plaintext is never
    stored or logged.
  * ``role_required`` reads the JWT identity/claims and rejects unauthorized
    roles with HTTP 403 (a missing/invalid token yields 401 via the JWT error
    handlers registered in the app factory).
"""

from functools import wraps
from typing import Callable, Iterable

import bcrypt
from flask import jsonify
from flask_jwt_extended import get_jwt, verify_jwt_in_request

# bcrypt work factor. Higher = slower = more resistant to brute force.
BCRYPT_ROUNDS = 12

# bcrypt only considers the first 72 bytes of the password; longer inputs raise
# in bcrypt 4.x. We defensively truncate to keep hashing deterministic.
_BCRYPT_MAX_BYTES = 72


def _encode(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(plain: str) -> str:
    """Return a bcrypt hash (cost 12) for ``plain``, encoded as UTF-8 text."""
    return bcrypt.hashpw(_encode(plain), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode(
        "utf-8"
    )


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time check of ``plain`` against a stored bcrypt ``hashed`` value.

    Returns ``False`` (rather than raising) if the stored hash is malformed, so
    callers can treat any failure as an authentication failure.
    """
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_encode(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _normalize_roles(roles: Iterable) -> set[str]:
    """Accept enum members or raw strings and return a set of role strings."""
    return {getattr(role, "value", role) for role in roles}


def role_required(allowed_roles: Iterable) -> Callable:
    """Decorator: allow the view only for the listed roles.

    Usage::

        @role_required([UserRole.admin, UserRole.head_office_staff])
        def view(): ...

    Behaviour:
      * No / invalid / expired / revoked token -> 401 (raised by
        ``verify_jwt_in_request`` and formatted by the JWT error handlers).
      * Valid token but role not permitted -> 403.
    """
    allowed = _normalize_roles(allowed_roles)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            role = get_jwt().get("role")
            if role not in allowed:
                return (
                    jsonify(
                        {
                            "error": "forbidden",
                            "message": "Your role is not permitted to access this resource.",
                        }
                    ),
                    403,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
