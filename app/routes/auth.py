"""Authentication routes: login, refresh, logout, me.

Also exposes ``GET /auth/admin-only``, a minimal role-protected route used to
verify the ``role_required`` decorator (401 without a token, 403 for the wrong
role) per the Phase 3 acceptance check.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required
from marshmallow import ValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.user_schema import login_schema, user_response_schema
from ..services.auth_service import (
    AuthError,
    authenticate,
    issue_tokens,
    refresh_access_token,
    revoke_current_token,
)
from ..utils.security import role_required

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/auth/login")
def login():
    """Exchange email + password for an access/refresh token pair."""
    try:
        data = login_schema.load(request.get_json(silent=True) or {})
    except ValidationError as err:
        return jsonify({"error": "validation_error", "messages": err.messages}), 422

    try:
        user = authenticate(data["email"], data["password"])
    except AuthError as err:
        return jsonify({"error": "authentication_failed", "message": err.message}), err.status

    access_token, refresh_token = issue_tokens(user)
    return (
        jsonify(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": user_response_schema.dump(user),
            }
        ),
        200,
    )


@auth_bp.post("/auth/refresh")
@jwt_required(refresh=True)
def refresh():
    """Issue a new access token given a valid refresh token."""
    new_access = refresh_access_token(get_jwt_identity(), get_jwt())
    return jsonify({"access_token": new_access}), 200


@auth_bp.post("/auth/logout")
@jwt_required(verify_type=False)
def logout():
    """Revoke the presented token (access or refresh) via the blocklist."""
    revoke_current_token()
    return jsonify({"message": "Token revoked."}), 200


@auth_bp.get("/auth/me")
@jwt_required()
def me():
    """Return the profile of the currently authenticated user."""
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "not_found", "message": "User no longer exists."}), 404
    return jsonify(user_response_schema.dump(user)), 200


@auth_bp.get("/auth/admin-only")
@role_required([UserRole.admin])
def admin_only():
    """Demonstration route guarded by ``role_required`` (admins only)."""
    return jsonify({"message": "Access granted: admin role verified."}), 200
