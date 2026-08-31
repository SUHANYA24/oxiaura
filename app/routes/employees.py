"""Employee KPI routes (BUILD_SPEC Phase 10).

Set a user's monthly target and list employees with their computed KPI
achievement. These are management views — restricted to admin / head office.
Row-scoping does not apply (this is an org-wide performance view).
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.employee_schema import (
    employee_target_response_schema,
    employee_target_schema,
)
from ..services import employee_service
from ..services.errors import ServiceError
from ..utils.security import role_required

employees_bp = Blueprint("employees", __name__)

_MANAGEMENT = [UserRole.admin, UserRole.head_office_staff]


def _service_error(err: ServiceError):
    return jsonify({"error": err.error, "message": err.message}), err.status


def _validation_error(err: SchemaValidationError):
    return jsonify({"error": "validation_error", "messages": err.messages}), 422


def _int_arg(name: str, default):
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@employees_bp.post("/employees/<int:user_id>/targets")
@role_required(_MANAGEMENT)
def set_target(user_id: int):
    """Create or update an employee's monthly KPI target."""
    try:
        data = employee_target_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        target = employee_service.set_target(user_id, data)
    except ServiceError as err:
        return _service_error(err)

    return jsonify(employee_target_response_schema.dump(target)), 200


@employees_bp.get("/employees")
@role_required(_MANAGEMENT)
def list_employees():
    """List employees with target/actuals and achievement % for a period."""
    rows = employee_service.list_employees_with_kpi(
        month=_int_arg("month", None), year=_int_arg("year", None)
    )
    return jsonify({"items": rows}), 200
