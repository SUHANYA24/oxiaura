"""Authentication service: credential check, token issuance, revocation.

Tokens carry the user's ``role`` and ``user_id`` as claims so authorization can
be enforced without a database lookup on every request (BUILD_SPEC section 3,
rule 2). The JWT ``sub`` (identity) is the stringified user id.
"""

from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
)

from ..models import User
from ..utils.security import verify_password
from ..utils.token_blocklist import add_to_blocklist


class AuthError(Exception):
    """Raised when authentication fails. Carries an HTTP status for the route."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.message = message
        self.status = status


def _claims_for(user: User) -> dict:
    """Additional JWT claims embedded in both access and refresh tokens."""
    return {"role": user.role.value, "user_id": user.id}


def authenticate(email: str, password: str) -> User:
    """Return the active user matching ``email``/``password`` or raise AuthError.

    The same generic message is returned for an unknown email and a wrong
    password so the endpoint does not leak which emails are registered.
    """
    user = User.query.filter_by(email=email).first()
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("Invalid email or password.", 401)
    if not user.is_active:
        raise AuthError("This account is disabled.", 403)
    return user


def issue_tokens(user: User) -> tuple[str, str]:
    """Create a fresh (access_token, refresh_token) pair for ``user``."""
    claims = _claims_for(user)
    identity = str(user.id)
    access_token = create_access_token(identity=identity, additional_claims=claims)
    refresh_token = create_refresh_token(identity=identity, additional_claims=claims)
    return access_token, refresh_token


def refresh_access_token(identity: str, claims: dict) -> str:
    """Mint a new access token from an already-verified refresh token.

    ``claims`` is the decoded refresh-token payload; role/user_id are carried
    forward so the new access token is self-describing.
    """
    forwarded = {"role": claims.get("role"), "user_id": claims.get("user_id")}
    return create_access_token(identity=identity, additional_claims=forwarded)


def revoke_current_token() -> None:
    """Add the currently presented token's ``jti`` to the blocklist (logout)."""
    payload = get_jwt()
    add_to_blocklist(payload["jti"], payload.get("exp"))
