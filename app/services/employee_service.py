"""Employee KPI service (BUILD_SPEC Phase 10).

Owns monthly targets and their live actuals. Targets are upserted per
(user, month, year); actuals are incremented as the business events that count
toward them occur — a customer registration bumps ``actual_customers`` and a
generated agreement bumps ``actual_revenue`` for the responsible rep.

The ``record_*`` helpers add to / mutate the current transaction but do **not**
commit, so they join the same unit of work as the event that triggered them
(mirroring :mod:`notification_service`). Business logic only — no Flask HTTP
types leak out.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from ..extensions import db
from ..models import EmployeeTarget, User
from .errors import NotFoundError


def _get_or_create_target(user_id: int, when: datetime | None = None) -> EmployeeTarget:
    """Return the target row for ``user_id`` in ``when``'s month/year, creating
    a zero-valued one if none exists so actuals always have somewhere to land."""
    when = when or datetime.utcnow()
    target = EmployeeTarget.query.filter_by(
        user_id=user_id, month=when.month, year=when.year
    ).first()
    if target is None:
        target = EmployeeTarget(
            user_id=user_id,
            month=when.month,
            year=when.year,
            target_customers=0,
            actual_customers=0,
            target_revenue=Decimal("0"),
            actual_revenue=Decimal("0"),
        )
        db.session.add(target)
    return target


def record_customer_registered(user_id: int, when: datetime | None = None) -> None:
    """Increment a rep's ``actual_customers`` for the event's month (no commit)."""
    target = _get_or_create_target(user_id, when)
    target.actual_customers = (target.actual_customers or 0) + 1


def record_revenue(user_id: int, amount, when: datetime | None = None) -> None:
    """Add ``amount`` to a rep's ``actual_revenue`` for the month (no commit)."""
    target = _get_or_create_target(user_id, when)
    target.actual_revenue = (target.actual_revenue or Decimal("0")) + Decimal(str(amount))


def set_target(user_id: int, data: dict) -> EmployeeTarget:
    """Create or update a user's monthly target, preserving existing actuals."""
    user = db.session.get(User, user_id)
    if user is None:
        raise NotFoundError("Employee not found.")

    target = EmployeeTarget.query.filter_by(
        user_id=user_id, month=data["month"], year=data["year"]
    ).first()
    if target is None:
        target = EmployeeTarget(user_id=user_id, month=data["month"], year=data["year"])
        db.session.add(target)

    target.target_customers = data["target_customers"]
    target.target_revenue = data["target_revenue"]
    db.session.commit()
    return target


def _achievement(actual, target) -> float | None:
    """Percentage of target achieved, or ``None`` when no target was set."""
    target = Decimal(str(target or 0))
    if target == 0:
        return None
    return round(float(Decimal(str(actual or 0)) / target * 100), 2)


def list_employees_with_kpi(*, month: int | None = None, year: int | None = None) -> list[dict]:
    """List every user with their target/actuals and achievement % for a period.

    Defaults to the current month/year. Users without a target for the period
    report zero actuals and ``None`` achievement.
    """
    now = datetime.utcnow()
    month = month or now.month
    year = year or now.year

    targets = {
        t.user_id: t
        for t in EmployeeTarget.query.filter_by(month=month, year=year).all()
    }

    rows = []
    for user in User.query.order_by(User.id).all():
        target = targets.get(user.id)
        target_customers = target.target_customers if target else 0
        actual_customers = target.actual_customers if target else 0
        target_revenue = target.target_revenue if target else Decimal("0")
        actual_revenue = target.actual_revenue if target else Decimal("0")
        rows.append(
            {
                "user_id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role.value,
                "branch_id": user.branch_id,
                "is_active": user.is_active,
                "month": month,
                "year": year,
                "target_customers": target_customers,
                "actual_customers": actual_customers,
                "target_revenue": f"{Decimal(str(target_revenue)):.2f}",
                "actual_revenue": f"{Decimal(str(actual_revenue)):.2f}",
                "customer_achievement_pct": _achievement(actual_customers, target_customers),
                "revenue_achievement_pct": _achievement(actual_revenue, target_revenue),
            }
        )
    return rows
