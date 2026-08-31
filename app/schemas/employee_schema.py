"""Schemas for employee KPI targets (BUILD_SPEC Phase 10).

The input schema rejects unknown fields. Actuals (``actual_customers`` /
``actual_revenue``) are server-owned — they are incremented as customers are
registered and agreements generated, never set by the client.
"""

from marshmallow import EXCLUDE, RAISE, Schema, fields, validate


class EmployeeTargetSchema(Schema):
    """Validates ``POST /employees/{id}/targets``."""

    class Meta:
        unknown = RAISE

    month = fields.Integer(required=True, validate=validate.Range(min=1, max=12))
    year = fields.Integer(required=True, validate=validate.Range(min=2000, max=2100))
    target_customers = fields.Integer(
        required=True, validate=validate.Range(min=0)
    )
    target_revenue = fields.Decimal(
        required=True, as_string=True, validate=validate.Range(min=0)
    )


class EmployeeTargetResponseSchema(Schema):
    """Representation of a stored monthly target with its actuals."""

    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(dump_only=True)
    user_id = fields.Integer(dump_only=True)
    month = fields.Integer(dump_only=True)
    year = fields.Integer(dump_only=True)
    target_customers = fields.Integer(dump_only=True)
    actual_customers = fields.Integer(dump_only=True)
    target_revenue = fields.Decimal(as_string=True, dump_only=True)
    actual_revenue = fields.Decimal(as_string=True, dump_only=True)


employee_target_schema = EmployeeTargetSchema()
employee_target_response_schema = EmployeeTargetResponseSchema()
