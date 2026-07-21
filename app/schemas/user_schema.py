"""Schemas for authentication and user serialization."""

from marshmallow import EXCLUDE, RAISE, Schema, fields, validate


class LoginSchema(Schema):
    """Validates the ``POST /auth/login`` body. Rejects unknown fields."""

    class Meta:
        unknown = RAISE

    email = fields.Email(required=True)
    password = fields.String(required=True, validate=validate.Length(min=1))


class UserResponseSchema(Schema):
    """Safe public representation of a user — never exposes ``password_hash``."""

    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(dump_only=True)
    full_name = fields.String(dump_only=True)
    email = fields.Email(dump_only=True)
    role = fields.Function(lambda user: user.role.value, dump_only=True)
    branch_id = fields.Integer(dump_only=True, allow_none=True)
    is_active = fields.Boolean(dump_only=True)
    created_at = fields.DateTime(dump_only=True)


login_schema = LoginSchema()
user_response_schema = UserResponseSchema()
