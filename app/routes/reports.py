"""Reporting routes (BUILD_SPEC Phase 10).

A management dashboard of aggregate stats and a filtered CSV export. Both are
restricted to admin / head office.
"""

from datetime import datetime

from flask import Blueprint, Response, jsonify, request

from ..models import UserRole
from ..services import report_service
from ..utils.security import role_required

reports_bp = Blueprint("reports", __name__)

_MANAGEMENT = [UserRole.admin, UserRole.head_office_staff]


def _int_arg(name: str):
    raw = request.args.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _date_arg(name: str):
    """Parse an ISO date/datetime query arg; raise ValueError on bad input."""
    raw = request.args.get(name)
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@reports_bp.get("/reports/dashboard")
@role_required(_MANAGEMENT)
def dashboard():
    """Aggregate stats: customers, agreements, fraud, and revenue by branch."""
    return jsonify(report_service.dashboard()), 200


@reports_bp.get("/reports/export")
@role_required(_MANAGEMENT)
def export():
    """Download a filtered customer report as CSV."""
    try:
        date_from = _date_arg("date_from")
        date_to = _date_arg("date_to")
    except ValueError:
        return (
            jsonify(
                {
                    "error": "validation_error",
                    "message": "date_from/date_to must be ISO-8601 dates.",
                }
            ),
            422,
        )

    csv_text = report_service.export_customers_csv(
        date_from=date_from,
        date_to=date_to,
        branch_id=_int_arg("branch_id"),
        rep_id=_int_arg("rep"),
    )
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=customers-export.csv"},
    )
