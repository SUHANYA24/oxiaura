"""Schemas for proposal submit / advance / response payloads (BUILD_SPEC Phase 9).

Input schemas reject unknown fields (section 3, rule 4). Server-owned fields
(``id``, ``sales_rep_id``, ``workflow_status``, ``submitted_at``) are never
accepted from the client: ``sales_rep_id`` is derived from the customer's
assigned rep, ``workflow_status`` starts at ``submitted`` and only ever changes
through the advance endpoint's state machine.
"""

from marshmallow import EXCLUDE, RAISE, Schema, fields, validate

from ..models import ProposalWorkflowStatus


class ProposalCreateSchema(Schema):
    """Validates ``POST /proposals``."""

    class Meta:
        unknown = RAISE

    customer_id = fields.Integer(required=True)
    proposed_amount = fields.Decimal(
        required=True,
        as_string=True,
        validate=validate.Range(min=0, min_inclusive=False),
    )
    product_type = fields.String(
        required=False, allow_none=True, validate=validate.Length(max=100)
    )
    notes = fields.String(required=False, allow_none=True)


class ProposalAdvanceSchema(Schema):
    """Validates ``PUT /proposals/{id}/advance``.

    ``decision`` is only meaningful at the ``ho_review`` stage, where the
    proposal branches to ``approved`` or ``rejected``; linear transitions
    ignore it. The service enforces when it is required.
    """

    class Meta:
        unknown = RAISE

    decision = fields.String(
        required=False,
        allow_none=True,
        validate=validate.OneOf(["approved", "rejected"]),
    )


class _CustomerSummarySchema(Schema):
    """Minimal customer representation embedded in proposal detail."""

    id = fields.Integer(dump_only=True)
    customer_code = fields.String(dump_only=True)
    full_name = fields.String(dump_only=True)


class ProposalResponseSchema(Schema):
    """List/summary representation of a proposal."""

    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(dump_only=True)
    customer_id = fields.Integer(dump_only=True)
    sales_rep_id = fields.Integer(dump_only=True)
    proposed_amount = fields.Decimal(as_string=True, dump_only=True)
    product_type = fields.String(dump_only=True, allow_none=True)
    workflow_status = fields.Enum(ProposalWorkflowStatus, by_value=True, dump_only=True)
    notes = fields.String(dump_only=True, allow_none=True)
    submitted_at = fields.DateTime(dump_only=True)


class ProposalDetailSchema(ProposalResponseSchema):
    """Detail representation: the proposal plus its customer summary."""

    customer = fields.Nested(_CustomerSummarySchema, dump_only=True)


# Singletons for reuse in routes.
proposal_create_schema = ProposalCreateSchema()
proposal_advance_schema = ProposalAdvanceSchema()
proposal_response_schema = ProposalResponseSchema()
proposal_detail_schema = ProposalDetailSchema()
proposals_response_schema = ProposalResponseSchema(many=True)
