"""Proposal routes (BUILD_SPEC Phase 9).

Submit a proposal, list/filter by status (scoped to the caller), view detail,
and advance the workflow one stage at a time. Roles are gated here; row-level
ownership and the per-transition role rules are enforced in the service.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.proposal_schema import (
    proposal_advance_schema,
    proposal_create_schema,
    proposal_detail_schema,
    proposal_response_schema,
    proposals_response_schema,
)
from ..services import proposal_service
from ..services.errors import ServiceError
from ..utils.security import role_required

proposals_bp = Blueprint("proposals", __name__)

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


@proposals_bp.post("/proposals")
@role_required(_ALL_ROLES)
def create_proposal():
    """Submit a new proposal (starts in the ``submitted`` stage)."""
    try:
        data = proposal_create_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        proposal = proposal_service.create_proposal(data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(proposal_response_schema.dump(proposal)), 201


@proposals_bp.get("/proposals")
@role_required(_ALL_ROLES)
def list_proposals():
    """Paginated list of proposals, optionally filtered by workflow status."""
    try:
        result = proposal_service.list_proposals(
            _current_user(),
            page=_int_arg("page", 1),
            per_page=_int_arg("per_page", 20),
            status=request.args.get("status"),
        )
    except ServiceError as err:
        return _service_error(err)

    return (
        jsonify(
            {
                "items": proposals_response_schema.dump(result.items),
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


@proposals_bp.get("/proposals/<int:proposal_id>")
@role_required(_ALL_ROLES)
def get_proposal(proposal_id: int):
    """Proposal detail including the customer summary."""
    try:
        proposal = proposal_service.get_proposal(proposal_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(proposal_detail_schema.dump(proposal)), 200


@proposals_bp.put("/proposals/<int:proposal_id>/advance")
@role_required(_ALL_ROLES)
def advance_proposal(proposal_id: int):
    """Advance a proposal one workflow stage.

    A ``decision`` of ``approved``/``rejected`` is required only when the
    proposal is under head-office review; the service enforces the role
    permitted for each transition.
    """
    try:
        data = proposal_advance_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        proposal = proposal_service.advance_proposal(
            proposal_id, _current_user(), decision=data.get("decision")
        )
    except ServiceError as err:
        return _service_error(err)

    return jsonify(proposal_response_schema.dump(proposal)), 200
