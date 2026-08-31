"""Schemas for proposal submit / update / advance / response payloads (BUILD_SPEC Phase 9).

Input schemas reject unknown fields (section 3, rule 4). Server-owned fields
(``id``, ``sales_rep_id``, ``workflow_status``, ``submitted_at``) are never
accepted from the client: ``sales_rep_id`` is derived from the customer's
assigned rep, ``workflow_status`` starts at ``submitted`` and only ever changes
through the advance endpoint's state machine.

``product_type`` is also server-owned: it is a snapshot of the selected
product's name, taken when the proposal is submitted or its product changed.
Clients choose a product by ``product_id`` and never name it directly.
"""

from marshmallow import EXCLUDE, RAISE, Schema, fields, validate

from ..models import ProductCategory, ProposalWorkflowStatus


class ProposalCreateSchema(Schema):
    """Validates ``POST /proposals``."""

    class Meta:
        unknown = RAISE

    customer_id = fields.Integer(required=True)
    product_id = fields.Integer(required=True)
    proposed_amount = fields.Decimal(
        required=True,
        as_string=True,
        validate=validate.Range(min=0, min_inclusive=False),
    )
    notes = fields.String(required=False, allow_none=True)


class ProposalUpdateSchema(Schema):
    """Validates ``PUT /proposals/{id}``. All fields optional (partial update).

    Only the commercial terms of a proposal are editable, and only while it is
    early in the workflow — the service rejects an edit once head office has it.
    ``customer_id`` is deliberately absent: re-pointing a proposal at a different
    customer would sidestep the rep-ownership check made at submission.
    """

    class Meta:
        unknown = RAISE

    product_id = fields.Integer()
    proposed_amount = fields.Decimal(
        as_string=True, validate=validate.Range(min=0, min_inclusive=False)
    )
    notes = fields.String(allow_none=True)


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


class _ProductSummarySchema(Schema):
    """Minimal product representation embedded in proposal detail.

    Reflects the product's *current* state. The proposal's own ``product_type``
    is the snapshot taken at submission, so the two differ once an admin edits
    the product — which is exactly what makes the history readable.
    """

    id = fields.Integer(dump_only=True)
    product_code = fields.String(dump_only=True)
    name = fields.String(dump_only=True)
    category = fields.Enum(ProductCategory, by_value=True, dump_only=True)
    duration_months = fields.Integer(dump_only=True)
    interest_rate = fields.Float(dump_only=True)
    is_active = fields.Boolean(dump_only=True)


class ProposalResponseSchema(Schema):
    """List/summary representation of a proposal."""

    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(dump_only=True)
    customer_id = fields.Integer(dump_only=True)
    sales_rep_id = fields.Integer(dump_only=True)
    product_id = fields.Integer(dump_only=True, allow_none=True)
    proposed_amount = fields.Decimal(as_string=True, dump_only=True)
    product_type = fields.String(dump_only=True, allow_none=True)
    workflow_status = fields.Enum(ProposalWorkflowStatus, by_value=True, dump_only=True)
    notes = fields.String(dump_only=True, allow_none=True)
    submitted_at = fields.DateTime(dump_only=True)


class ProposalDetailSchema(ProposalResponseSchema):
    """Detail representation: the proposal plus its customer and product."""

    customer = fields.Nested(_CustomerSummarySchema, dump_only=True)
    product = fields.Nested(_ProductSummarySchema, dump_only=True, allow_none=True)


# Singletons for reuse in routes.
proposal_create_schema = ProposalCreateSchema()
proposal_update_schema = ProposalUpdateSchema()
proposal_advance_schema = ProposalAdvanceSchema()
proposal_response_schema = ProposalResponseSchema()
proposal_detail_schema = ProposalDetailSchema()
proposals_response_schema = ProposalResponseSchema(many=True)
