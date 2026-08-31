"""Proposal service (BUILD_SPEC Phase 9).

A proposal moves through a linear workflow with a terminal branch::

    submitted --> rep_review --> ho_review --> approved
                                          \\--> rejected

Each transition is gated by role (only the correct role may perform it) and by
row-level ownership (a ``sales_rep`` only touches proposals for their own
customers). A notification is queued on every stage change. Business logic only
— no Flask HTTP types leak out; routes translate ``ServiceError`` to responses.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from ..extensions import db
from ..models import Proposal, ProposalWorkflowStatus, User, UserRole
from . import customer_service, notification_service, product_service
from .errors import ForbiddenError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)

_MAX_PER_PAGE = 100
_DEFAULT_PER_PAGE = 20

_S = ProposalWorkflowStatus
_TERMINAL = {_S.approved, _S.rejected}

# Stages at which the commercial terms may still be edited. Once head office has
# the proposal, changing the product or amount underneath their review would
# invalidate the decision they are making.
_EDITABLE = {_S.submitted, _S.rep_review}

# The non-branching transitions: current stage -> its single next stage.
_LINEAR_NEXT = {_S.submitted: _S.rep_review, _S.rep_review: _S.ho_review}

# Which role (besides ``admin``, who may perform any transition) is allowed to
# move a proposal along each (from -> to) edge.
_TRANSITION_ROLES = {
    (_S.submitted, _S.rep_review): {UserRole.sales_rep},
    (_S.rep_review, _S.ho_review): {UserRole.sales_rep},
    (_S.ho_review, _S.approved): {UserRole.head_office_staff},
    (_S.ho_review, _S.rejected): {UserRole.head_office_staff},
}


def create_proposal(data: dict, current_user: User) -> Proposal:
    """Submit a new proposal for a customer the caller may access.

    The owning ``sales_rep`` is derived from the customer's assignment, so a
    rep can only ever submit for their own customers (enforced by
    :func:`customer_service.get_customer`).

    ``product_id`` must name a live, on-sale catalog product. Its name is copied
    into ``product_type`` as a snapshot, so a later admin edit to the product
    does not rewrite what this proposal recorded.
    """
    customer = customer_service.get_customer(data["customer_id"], current_user)
    product = product_service.resolve_selectable_product(data["product_id"])

    proposal = Proposal(
        customer_id=customer.id,
        sales_rep_id=customer.assigned_rep_id,
        product_id=product.id,
        proposed_amount=data["proposed_amount"],
        product_type=product.name,
        notes=data.get("notes"),
        workflow_status=_S.submitted,
    )
    db.session.add(proposal)
    db.session.commit()
    logger.info(
        "Proposal %s submitted for customer %s (product %s) by user %s",
        proposal.id,
        customer.customer_code,
        product.product_code,
        current_user.id,
    )
    return proposal


def _scoped_query(current_user: User):
    """Base select for proposals visible to ``current_user``."""
    stmt = select(Proposal)
    if not customer_service._has_full_access(current_user):
        stmt = stmt.where(Proposal.sales_rep_id == current_user.id)
    return stmt


def list_proposals(
    current_user: User,
    *,
    page: int = 1,
    per_page: int = _DEFAULT_PER_PAGE,
    status: str | None = None,
):
    """Return a paginated list of proposals scoped to ``current_user``."""
    page = max(page, 1)
    per_page = min(max(per_page, 1), _MAX_PER_PAGE)

    stmt = _scoped_query(current_user)
    if status is not None:
        try:
            status_enum = _S(status)
        except ValueError:
            raise ValidationError(
                f"Invalid status '{status}'. Expected one of: "
                f"{', '.join(s.value for s in _S)}."
            )
        stmt = stmt.where(Proposal.workflow_status == status_enum)

    stmt = stmt.order_by(Proposal.id.desc())
    return db.paginate(stmt, page=page, per_page=per_page, error_out=False)


def get_proposal(proposal_id: int, current_user: User) -> Proposal:
    """Fetch one proposal, enforcing row-level ownership for sales reps."""
    proposal = db.session.get(Proposal, proposal_id)
    if proposal is None:
        raise NotFoundError("Proposal not found.")
    if (
        not customer_service._has_full_access(current_user)
        and proposal.sales_rep_id != current_user.id
    ):
        raise ForbiddenError("You do not have access to this proposal.")
    return proposal


def update_proposal(proposal_id: int, data: dict, current_user: User) -> Proposal:
    """Revise a proposal's commercial terms before head office reviews it.

    Row-level ownership is reused from :func:`get_proposal`, so a rep may only
    revise their own proposals. Changing ``product_id`` re-validates the product
    and re-takes the ``product_type`` snapshot.

    :raises ValidationError: the proposal has moved past ``rep_review``, or the
        chosen product is retired / off-sale.
    :raises ForbiddenError: the caller does not own this proposal.
    """
    proposal = get_proposal(proposal_id, current_user)

    if proposal.workflow_status not in _EDITABLE:
        raise ValidationError(
            f"A proposal under {proposal.workflow_status.value} can no longer be "
            f"edited. Editable stages: "
            f"{', '.join(s.value for s in (_S.submitted, _S.rep_review))}."
        )

    if "product_id" in data:
        product = product_service.resolve_selectable_product(data["product_id"])
        proposal.product_id = product.id
        # Re-snapshot: the proposal now records the newly chosen product.
        proposal.product_type = product.name

    for field in ("proposed_amount", "notes"):
        if field in data:
            setattr(proposal, field, data[field])

    db.session.commit()
    logger.info(
        "Proposal %s revised by user %s (fields: %s)",
        proposal.id,
        current_user.id,
        ", ".join(sorted(data)) or "none",
    )
    return proposal


def _resolve_target(current: ProposalWorkflowStatus, decision: str | None):
    """Determine the next stage, requiring a decision only at ``ho_review``."""
    if current == _S.ho_review:
        if decision not in ("approved", "rejected"):
            raise ValidationError(
                "A decision of 'approved' or 'rejected' is required to "
                "finalize a proposal under head-office review."
            )
        return _S(decision)
    return _LINEAR_NEXT[current]


def advance_proposal(
    proposal_id: int, current_user: User, *, decision: str | None = None
) -> Proposal:
    """Move a proposal to its next workflow stage.

    :raises ForbiddenError: the caller's role may not perform this transition.
    :raises ValidationError: the proposal is already terminal, or a decision is
        required (at ``ho_review``) but missing/invalid.
    """
    proposal = get_proposal(proposal_id, current_user)
    current = proposal.workflow_status
    if current in _TERMINAL:
        raise ValidationError(
            f"Proposal is already {current.value}; no further transitions are allowed."
        )

    target = _resolve_target(current, decision)

    allowed = _TRANSITION_ROLES.get((current, target), set())
    if current_user.role != UserRole.admin and current_user.role not in allowed:
        raise ForbiddenError(
            f"Your role may not advance a proposal from {current.value} to {target.value}."
        )

    proposal.workflow_status = target
    _notify_stage_change(proposal, current_user)
    db.session.commit()
    logger.info(
        "Proposal %s advanced %s -> %s by user %s",
        proposal.id,
        current.value,
        target.value,
        current_user.id,
    )
    return proposal


def _notify_stage_change(proposal: Proposal, actor: User) -> None:
    """Queue notifications for a stage change (joins the caller's transaction)."""
    status = proposal.workflow_status
    title = f"Proposal #{proposal.id} is now {status.value}"
    message = f"Proposal #{proposal.id} moved to '{status.value}' by user {actor.id}."

    # The owning rep is always informed of movement on their proposal.
    notification_service.notify_user(proposal.sales_rep_id, title, message)
    # When it reaches head-office review, alert the staff who must act on it.
    if status == _S.ho_review:
        notification_service.notify_staff(title, message)
