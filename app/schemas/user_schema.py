"""Schemas for authentication and user administration.

Input schemas reject unknown fields (BUILD_SPEC section 3, rule 4). Server-owned
fields (``id``, ``password_hash``, ``created_at``) are never accepted from the
client — passwords arrive as plaintext and are hashed by the service, and no
response schema exposes ``password_hash``.
"""

from marshmallow import EXCLUDE, RAISE, Schema, ValidationError, fields, validate

from ..models import UserRole

# Reused validators.
_full_name = validate.Length(min=1, max=150)
_email_length = validate.Length(max=150)

# BUILD_SPEC section 3 rule 1 mandates bcrypt cost 12 but states no password
# policy. These are the house rules for anything the API accepts; bcrypt only
# considers the first 72 bytes, so there is no point allowing more.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 72


def _validate_password_strength(value: str) -> None:
    """Require at least one letter and one digit alongside the length check."""
    if not any(char.isalpha() for char in value):
        raise ValidationError("Password must contain at least one letter.")
    if not any(char.isdigit() for char in value):
        raise ValidationError("Password must contain at least one digit.")


_password = [
    validate.Length(min=PASSWORD_MIN_LENGTH, max=PASSWORD_MAX_LENGTH),
    _validate_password_strength,
]


class LoginSchema(Schema):
    """Validates the ``POST /auth/login`` body. Rejects unknown fields."""

    class Meta:
        unknown = RAISE

    email = fields.Email(required=True)
    password = fields.String(required=True, validate=validate.Length(min=1))


class UserCreateSchema(Schema):
    """Validates ``POST /users``. The service hashes ``password`` and drops it."""

    class Meta:
        unknown = RAISE

    full_name = fields.String(required=True, validate=_full_name)
    email = fields.Email(required=True, validate=_email_length)
    password = fields.String(required=True, load_only=True, validate=_password)
    role = fields.Enum(UserRole, by_value=True, required=True)
    branch_id = fields.Integer(required=False, allow_none=True)
    # Accounts are active unless explicitly created disabled.
    is_active = fields.Boolean(required=False, load_default=True)


class UserUpdateSchema(Schema):
    """Validates ``PUT /users/{id}``. All fields optional (partial update).

    ``password`` is deliberately absent — it is changed only through the
    dedicated password endpoints so a credential change is never an incidental
    side effect of a profile edit.
    """

    class Meta:
        unknown = RAISE

    full_name = fields.String(validate=_full_name)
    email = fields.Email(validate=_email_length)
    role = fields.Enum(UserRole, by_value=True)
    branch_id = fields.Integer(allow_none=True)
    is_active = fields.Boolean()


class PasswordResetSchema(Schema):
    """Validates ``POST /users/{id}/reset-password`` (admin sets a password).

    No current password is required: the caller is an admin acting on another
    account, not the account holder.
    """

    class Meta:
        unknown = RAISE

    new_password = fields.String(required=True, load_only=True, validate=_password)


class PasswordChangeSchema(Schema):
    """Validates ``PUT /users/me/password`` (a user changes their own)."""

    class Meta:
        unknown = RAISE

    current_password = fields.String(
        required=True, load_only=True, validate=validate.Length(min=1)
    )
    new_password = fields.String(required=True, load_only=True, validate=_password)


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


class _BranchSummarySchema(Schema):
    """Minimal representation of the branch a user belongs to."""

    id = fields.Integer(dump_only=True)
    name = fields.String(dump_only=True)
    location = fields.String(dump_only=True, allow_none=True)


class UserDetailSchema(UserResponseSchema):
    """Detail representation: the user plus their resolved branch."""

    branch = fields.Nested(_BranchSummarySchema, dump_only=True, allow_none=True)


# Singletons for reuse in routes.
login_schema = LoginSchema()
user_create_schema = UserCreateSchema()
user_update_schema = UserUpdateSchema()
password_reset_schema = PasswordResetSchema()
password_change_schema = PasswordChangeSchema()
user_response_schema = UserResponseSchema()
user_detail_schema = UserDetailSchema()
users_response_schema = UserResponseSchema(many=True)
