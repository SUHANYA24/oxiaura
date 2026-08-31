"""User administration routes.

Flow per request: role guard -> Marshmallow schema -> service -> ORM -> schema
response, mirroring the customer module. There is no row-level scoping here —
this is an org-wide operator directory, so the decorators are the whole access
story: reads are open to management, every mutation is admin-only, and the one
self-service route (own password change) is open to any authenticated user.

Passwords are never returned or logged; ``DELETE`` deactivates rather than
deletes, because five tables reference ``users.id``.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.user_schema import (
    password_change_schema,
    password_reset_schema,
    user_create_schema,
    user_detail_schema,
    user_response_schema,
    user_update_schema,
    users_response_schema,
)
from ..services import user_service
from ..services.errors import ServiceError
from ..utils.security import role_required

users_bp = Blueprint("users", __name__)

_MANAGEMENT = [UserRole.admin, UserRole.head_office_staff]
_ALL_ROLES = [UserRole.admin, UserRole.head_office_staff, UserRole.sales_rep]


def _current_user() -> User:
    """Load the authenticated user from the JWT identity."""
    return db.session.get(User, int(get_jwt_identity()))


def _service_error(err: ServiceError):
    return jsonify({"error": err.error, "message": err.message}), err.status


def _validation_error(err: SchemaValidationError):
    return jsonify({"error": "validation_error", "messages": err.messages}), 422


def _int_arg(name: str, default):
    """Parse an integer query arg, falling back to ``default`` if absent/invalid."""
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _bool_arg(name: str, default=None):
    """Parse a boolean query arg. Anything unrecognized falls back to ``default``."""
    raw = request.args.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return default


@users_bp.get("/users")
@role_required(_MANAGEMENT)
def list_users():
    """Paginated, filterable operator directory (management only)."""
    try:
        result = user_service.list_users(
            page=_int_arg("page", 1),
            per_page=_int_arg("per_page", 20),
            role=request.args.get("role"),
            branch_id=_int_arg("branch_id", None),
            is_active=_bool_arg("is_active"),
            search=request.args.get("search"),
        )
    except ServiceError as err:
        return _service_error(err)

    return (
        jsonify(
            {
                "items": users_response_schema.dump(result.items),
                "pagination": {
                    "page": result.page,
                    "per_page": result.per_page,
                    "total": result.total,
                    "pages": result.pages,
                    "has_next": result.has_next,
                    "has_prev": result.has_prev,
                },
            }
        ),
        200,
    )


@users_bp.post("/users")
@role_required([UserRole.admin])
def create_user():
    """Create an operator account (admin only). Duplicate email -> 409."""
    try:
        data = user_create_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        user = user_service.create_user(data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(user_response_schema.dump(user)), 201


@users_bp.put("/users/me/password")
@jwt_required()
def change_own_password():
    """Change your own password. Requires the current password."""
    try:
        data = password_change_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        user_service.change_own_password(
            _current_user(), data["current_password"], data["new_password"]
        )
    except ServiceError as err:
        return _service_error(err)

    return jsonify({"message": "Password updated."}), 200


@users_bp.get("/users/<int:user_id>")
@role_required(_MANAGEMENT)
def get_user(user_id: int):
    """User detail including the resolved branch (management only)."""
    try:
        user = user_service.get_user(user_id)
    except ServiceError as err:
        return _service_error(err)

    return jsonify(user_detail_schema.dump(user)), 200


@users_bp.put("/users/<int:user_id>")
@role_required([UserRole.admin])
def update_user(user_id: int):
    """Partial update (admin only). Cannot change your own role or status."""
    try:
        data = user_update_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        user = user_service.update_user(user_id, data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(user_response_schema.dump(user)), 200


@users_bp.post("/users/<int:user_id>/reset-password")
@role_required([UserRole.admin])
def reset_password(user_id: int):
    """Set another user's password (admin only)."""
    try:
        data = password_reset_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        user_service.reset_password(user_id, data["new_password"], _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify({"message": "Password reset."}), 200


@users_bp.delete("/users/<int:user_id>")
@role_required([UserRole.admin])
def deactivate_user(user_id: int):
    """Deactivate an account (admin only). Rows are never hard-deleted."""
    try:
        user = user_service.deactivate_user(user_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return (
        jsonify(
            {
                "message": "User deactivated.",
                "user": user_response_schema.dump(user),
            }
        ),
        200,
    )
