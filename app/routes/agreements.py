"""Agreement routes (BUILD_SPEC Phase 8).

Generate an agreement (PDF + signed QR), list/detail scoped to the caller,
download the PDF, and a **public** authenticity check. Roles are gated here;
row-level ownership (a sales_rep only touching their own customers' agreements)
is enforced in the service, matching the customer/document slices.
"""

from flask import Blueprint, jsonify, request, send_file
from flask_jwt_extended import get_jwt_identity
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.agreement_schema import (
    agreement_create_schema,
    agreement_detail_schema,
    agreement_response_schema,
    agreements_response_schema,
)
from ..services import agreement_service
from ..services.errors import ServiceError
from ..utils.security import role_required

agreements_bp = Blueprint("agreements", __name__)

_ALL_ROLES = [UserRole.admin, UserRole.head_office_staff, UserRole.sales_rep]


def _current_user() -> User:
    return db.session.get(User, int(get_jwt_identity()))


def _service_error(err: ServiceError):
    return jsonify({"error": err.error, "message": err.message}), err.status


def _validation_error(err: SchemaValidationError):
    return jsonify({"error": "validation_error", "messages": err.messages}), 422


def _int_arg(name: str, default: int) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@agreements_bp.post("/agreements/generate")
@role_required(_ALL_ROLES)
def generate_agreement():
    """Create an agreement, render its PDF, and mint a signed QR token."""
    try:
        data = agreement_create_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        agreement = agreement_service.generate_agreement(data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(agreement_response_schema.dump(agreement)), 201


@agreements_bp.get("/agreements")
@role_required(_ALL_ROLES)
def list_agreements():
    """Paginated list of agreements, scoped to the caller."""
    result = agreement_service.list_agreements(
        _current_user(), page=_int_arg("page", 1), per_page=_int_arg("per_page", 20)
    )
    return (
        jsonify(
            {
                "items": agreements_response_schema.dump(result.items),
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


@agreements_bp.get("/agreements/<int:agreement_id>")
@role_required(_ALL_ROLES)
def get_agreement(agreement_id: int):
    """Agreement detail including the customer summary."""
    try:
        agreement = agreement_service.get_agreement(agreement_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(agreement_detail_schema.dump(agreement)), 200


@agreements_bp.get("/agreements/<int:agreement_id>/pdf")
@role_required(_ALL_ROLES)
def download_pdf(agreement_id: int):
    """Download the generated agreement PDF."""
    try:
        path = agreement_service.get_pdf_path(agreement_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return send_file(
        path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"agreement-{agreement_id}.pdf",
    )


@agreements_bp.get("/verify/<token>")
def verify_agreement(token: str):
    """**Public** authenticity check — no authentication required.

    Returns whether the QR token was issued by us (valid signature + a matching
    stored agreement). A tampered token returns ``valid: False``.
    """
    result = agreement_service.verify_agreement(token)
    status = 200 if result.get("valid") else 404
    return jsonify(result), status
