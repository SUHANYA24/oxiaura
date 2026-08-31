"""User service — administration of operator accounts (the ``users`` table).

Access rules:
  * ``admin`` may create, update, deactivate, and reset the password of any user.
  * ``head_office_staff`` gets read-only visibility (gated at the route).
  * Every authenticated user may change their own password.

Users are **never hard-deleted**. Five tables reference ``users.id`` — customers
(``assigned_rep_id``), agreements (``created_by``), proposals (``sales_rep_id``),
employee targets, and notifications — so removing a row would orphan business
history. ``DELETE /users/{id}`` clears ``is_active`` instead, which is also what
blocks login (``auth_service.authenticate`` returns 403 for a disabled account).

Two guards keep the admin role from being locked out of the system:
  * a caller cannot change their own ``role`` or ``is_active`` (no self-demotion,
    no self-deactivation — that is what another admin is for);
  * the last active admin cannot be demoted or deactivated by anyone.

All database access goes through the ORM — no raw SQL (section 3, rule 6).
"""

import logging

from sqlalchemy import func, or_, select

from ..extensions import db
from ..models import Branch, User, UserRole
from ..utils.security import hash_password, verify_password
from .errors import ConflictError, ForbiddenError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)

_MAX_PER_PAGE = 100
_DEFAULT_PER_PAGE = 20

# Fields a caller may never change on their own account.
_SELF_PROTECTED_FIELDS = ("role", "is_active")


def _normalize_email(email: str) -> str:
    """Lowercase and trim so uniqueness is effectively case-insensitive."""
    return email.strip().lower()


def _email_taken(email: str, *, exclude_id: int | None = None) -> bool:
    """True if another user row already holds this (normalized) email."""
    query = User.query.filter(User.email == email)
    if exclude_id is not None:
        query = query.filter(User.id != exclude_id)
    return db.session.query(query.exists()).scalar()


def _resolve_branch_id(branch_id: int | None) -> int | None:
    """Validate an optional branch reference, returning it unchanged."""
    if branch_id is None:
        return None
    if db.session.get(Branch, branch_id) is None:
        raise ValidationError("branch_id does not reference an existing branch.")
    return branch_id


def _other_active_admins(user_id: int) -> int:
    """Count active admins other than ``user_id``."""
    return (
        db.session.query(func.count(User.id))
        .filter(
            User.role == UserRole.admin,
            User.is_active.is_(True),
            User.id != user_id,
        )
        .scalar()
    )


def _assert_not_last_admin(user: User) -> None:
    """Refuse a change that would leave the system with no active admin."""
    is_active_admin = user.role == UserRole.admin and user.is_active
    if is_active_admin and _other_active_admins(user.id) == 0:
        raise ConflictError(
            "This is the last active admin. Promote another admin first."
        )


def _assert_not_self(user: User, current_user: User, action: str) -> None:
    """Block privilege changes a caller would be making to their own account."""
    if user.id == current_user.id:
        raise ForbiddenError(
            f"You cannot {action} your own account. Ask another admin to do it."
        )


def get_user(user_id: int) -> User:
    """Fetch a single user by id. Route restricts this to management roles."""
    user = db.session.get(User, user_id)
    if user is None:
        raise NotFoundError("User not found.")
    return user


def create_user(data: dict, current_user: User) -> User:
    """Create an operator account. Rejects a duplicate email with 409."""
    email = _normalize_email(data["email"])
    if _email_taken(email):
        raise ConflictError(f"A user with email {email} already exists.")

    user = User(
        full_name=data["full_name"],
        email=email,
        password_hash=hash_password(data["password"]),
        role=data["role"],
        branch_id=_resolve_branch_id(data.get("branch_id")),
        is_active=data.get("is_active", True),
    )
    db.session.add(user)
    db.session.commit()

    logger.info(
        "user_created id=%s email=%s role=%s by_user=%s",
        user.id,
        user.email,
        user.role.value,
        current_user.id,
    )
    return user


def list_users(
    *,
    page: int = 1,
    per_page: int = _DEFAULT_PER_PAGE,
    role: str | None = None,
    branch_id: int | None = None,
    is_active: bool | None = None,
    search: str | None = None,
):
    """Return a paginated, filtered list of users.

    This is an org-wide directory view with no row-level scoping — the route
    restricts it to management roles.
    """
    page = max(page, 1)
    per_page = min(max(per_page, 1), _MAX_PER_PAGE)

    stmt = select(User)

    if role is not None:
        try:
            role_enum = UserRole(role)
        except ValueError:
            raise ValidationError(
                f"Invalid role '{role}'. Expected one of: "
                f"{', '.join(r.value for r in UserRole)}."
            )
        stmt = stmt.where(User.role == role_enum)

    if branch_id is not None:
        stmt = stmt.where(User.branch_id == branch_id)

    if is_active is not None:
        stmt = stmt.where(User.is_active.is_(is_active))

    if search:
        term = f"%{search.strip()}%"
        stmt = stmt.where(or_(User.full_name.ilike(term), User.email.ilike(term)))

    stmt = stmt.order_by(User.id.desc())
    return db.paginate(stmt, page=page, per_page=per_page, error_out=False)


def update_user(user_id: int, data: dict, current_user: User) -> User:
    """Apply a partial update, re-checking email uniqueness and admin guards."""
    user = get_user(user_id)

    # Role and activation are privilege changes: never on your own account, and
    # never if they would strip the system of its last active admin.
    if any(field in data for field in _SELF_PROTECTED_FIELDS):
        changing = {
            field: data[field]
            for field in _SELF_PROTECTED_FIELDS
            if field in data and data[field] != getattr(user, field)
        }
        if changing:
            _assert_not_self(user, current_user, "change the role or status of")
            demoted = changing.get("role", user.role) != UserRole.admin
            deactivated = changing.get("is_active", user.is_active) is False
            if demoted or deactivated:
                _assert_not_last_admin(user)

    if "email" in data:
        email = _normalize_email(data["email"])
        if email != user.email and _email_taken(email, exclude_id=user.id):
            raise ConflictError(f"A user with email {email} already exists.")
        user.email = email

    if "branch_id" in data:
        user.branch_id = _resolve_branch_id(data["branch_id"])

    for field in ("full_name", "role", "is_active"):
        if field in data:
            setattr(user, field, data[field])

    db.session.commit()

    logger.info(
        "user_updated id=%s fields=%s by_user=%s",
        user.id,
        ",".join(sorted(data)),
        current_user.id,
    )
    return user


def deactivate_user(user_id: int, current_user: User) -> User:
    """Disable an account (soft delete). Idempotent — re-disabling is a no-op."""
    user = get_user(user_id)
    if not user.is_active:
        return user

    _assert_not_self(user, current_user, "deactivate")
    _assert_not_last_admin(user)

    user.is_active = False
    db.session.commit()

    logger.info("user_deactivated id=%s by_user=%s", user.id, current_user.id)
    return user


def reset_password(user_id: int, new_password: str, current_user: User) -> User:
    """Set a user's password on their behalf. Route restricts this to admins."""
    user = get_user(user_id)
    user.password_hash = hash_password(new_password)
    db.session.commit()

    logger.info("user_password_reset id=%s by_user=%s", user.id, current_user.id)
    return user


def change_own_password(
    current_user: User, current_password: str, new_password: str
) -> User:
    """Change the caller's own password after verifying the current one."""
    if not verify_password(current_password, current_user.password_hash):
        raise ForbiddenError("Current password is incorrect.")
    if verify_password(new_password, current_user.password_hash):
        raise ValidationError("New password must differ from the current one.")

    current_user.password_hash = hash_password(new_password)
    db.session.commit()

    logger.info("user_password_changed id=%s", current_user.id)
    return current_user
