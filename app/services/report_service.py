"""Reporting service (BUILD_SPEC Phase 10).

Read-only aggregate reporting for the management dashboard plus a filtered CSV
export. All aggregation goes through the ORM / SQLAlchemy expression API — no
raw SQL (section 3, rule 6). Business logic only; the route wraps the returned
CSV text in an HTTP download response.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from decimal import Decimal

from sqlalchemy import func, select

from ..extensions import db
from ..models import (
    Agreement,
    AgreementStatus,
    Branch,
    Customer,
    CustomerStatus,
    Document,
    FraudLog,
    User,
)


def _money(value) -> str:
    return f"{Decimal(str(value or 0)):.2f}"


def dashboard() -> dict:
    """Aggregate stats: customer counts, agreement totals, fraud stats, revenue."""
    active_customers = Customer.is_deleted.is_(False)

    total_customers = db.session.query(func.count(Customer.id)).filter(
        active_customers
    ).scalar()
    customers_by_status = {s.value: 0 for s in CustomerStatus}
    for status, count in (
        db.session.query(Customer.status, func.count(Customer.id))
        .filter(active_customers)
        .group_by(Customer.status)
        .all()
    ):
        customers_by_status[status.value] = count

    total_agreements = db.session.query(func.count(Agreement.id)).scalar()
    agreements_by_status = {s.value: 0 for s in AgreementStatus}
    for status, count in (
        db.session.query(Agreement.status, func.count(Agreement.id))
        .group_by(Agreement.status)
        .all()
    ):
        agreements_by_status[status.value] = count
    total_investment = db.session.query(
        func.coalesce(func.sum(Agreement.investment_amount), 0)
    ).scalar()

    documents_total = db.session.query(func.count(Document.id)).scalar()
    fraud_checked = db.session.query(func.count(FraudLog.id)).scalar()
    fraud_flagged = (
        db.session.query(func.count(FraudLog.id))
        .filter(FraudLog.is_flagged.is_(True))
        .scalar()
    )

    branch_rows = (
        db.session.query(
            Branch.id,
            Branch.name,
            func.coalesce(func.sum(Agreement.investment_amount), 0),
        )
        .select_from(Agreement)
        .join(Customer, Agreement.customer_id == Customer.id)
        .join(User, Customer.assigned_rep_id == User.id)
        .join(Branch, User.branch_id == Branch.id)
        .group_by(Branch.id, Branch.name)
        .all()
    )
    revenue_by_branch = [
        {"branch_id": bid, "branch_name": name, "total_investment": _money(total)}
        for bid, name, total in branch_rows
    ]

    return {
        "customers": {"total": total_customers, "by_status": customers_by_status},
        "agreements": {
            "total": total_agreements,
            "by_status": agreements_by_status,
            "total_investment": _money(total_investment),
        },
        "fraud": {
            "documents_total": documents_total,
            "checked": fraud_checked,
            "flagged": fraud_flagged,
        },
        "revenue_by_branch": revenue_by_branch,
    }


_EXPORT_FIELDS = [
    "customer_code",
    "full_name",
    "nic_number",
    "status",
    "assigned_rep_id",
    "branch_id",
    "registered_at",
]


def export_customers_csv(
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    branch_id: int | None = None,
    rep_id: int | None = None,
) -> str:
    """Return a CSV export of customers, filtered by date range / branch / rep."""
    stmt = select(Customer).where(Customer.is_deleted.is_(False))
    if date_from is not None:
        stmt = stmt.where(Customer.registered_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(Customer.registered_at <= date_to)
    if rep_id is not None:
        stmt = stmt.where(Customer.assigned_rep_id == rep_id)
    if branch_id is not None:
        stmt = stmt.join(User, Customer.assigned_rep_id == User.id).where(
            User.branch_id == branch_id
        )
    stmt = stmt.order_by(Customer.id)

    customers = db.session.execute(stmt).scalars().all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_EXPORT_FIELDS)
    for c in customers:
        writer.writerow(
            [
                c.customer_code,
                c.full_name,
                c.nic_number,
                c.status.value,
                c.assigned_rep_id,
                c.assigned_rep.branch_id if c.assigned_rep else "",
                c.registered_at.isoformat() if c.registered_at else "",
            ]
        )
    return buffer.getvalue()
