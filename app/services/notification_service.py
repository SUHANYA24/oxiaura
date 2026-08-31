"""Notification service — create in-app notifications for users.

Business logic only; no Flask HTTP types. Callers are responsible for the
surrounding transaction (these helpers ``add`` to the session but do not commit)
so notifications join the same unit of work as the event that triggered them.
"""

from __future__ import annotations

from ..extensions import db
from ..models import Notification, User, UserRole


def notify_user(user_id: int, title: str, message: str | None = None) -> Notification:
    """Queue a notification for a single user (not committed)."""
    notification = Notification(user_id=user_id, title=title, message=message)
    db.session.add(notification)
    return notification


def notify_roles(roles, title: str, message: str | None = None) -> list[Notification]:
    """Queue a notification for every active user holding one of ``roles``."""
    role_values = [getattr(r, "value", r) for r in roles]
    recipients = User.query.filter(
        User.role.in_(role_values), User.is_active.is_(True)
    ).all()
    return [notify_user(user.id, title, message) for user in recipients]


def notify_staff(title: str, message: str | None = None) -> list[Notification]:
    """Notify the roles responsible for verification (admin + head office)."""
    return notify_roles([UserRole.admin, UserRole.head_office_staff], title, message)
