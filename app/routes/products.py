"""Product catalog routes.

Flow per request: role guard -> Marshmallow schema -> service -> ORM -> schema
response, mirroring the customer and user modules. There is no row-level scoping
here — the catalog is org-wide reference data, so the decorators are the whole
access story:

  * reads are open to every authenticated role, because a sales rep must be able
    to pick a product when submitting a proposal;
  * every mutation is admin-only.

``DELETE`` retires rather than removes, because ``proposals.product_id``
references ``products.id``.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.product_schema import (
    product_create_schema,
    product_detail_schema,
    product_response_schema,
    product_update_schema,
    products_response_schema,
)
from ..services import product_service
from ..services.errors import ServiceError
from ..utils.security import role_required

products_bp = Blueprint("products", __name__)

_ALL_ROLES = [UserRole.admin, UserRole.head_office_staff, UserRole.sales_rep]


def _current_user() -> User:
    """Load the authenticated user from the JWT identity."""
    return db.session.get(User, int(get_jwt_identity()))


def _service_error(err: ServiceError):
    return jsonify({"error": err.error, "message": err.message}), err.status


def _validation_error(err: SchemaValidationError):
    return jsonify({"error": "validation_error", "messages": err.messages}), 422


def _int_arg(name: str, default):
    """Parse an integer query arg, falling back to ``default`` if absent/invalid."""
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _bool_arg(name: str, default=None):
    """Parse a boolean query arg. Anything unrecognized falls back to ``default``."""
    raw = request.args.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return default


@products_bp.get("/products")
@role_required(_ALL_ROLES)
def list_products():
    """Paginated, filterable catalog (any authenticated role)."""
    try:
        result = product_service.list_products(
            page=_int_arg("page", 1),
            per_page=_int_arg("per_page", 20),
            category=request.args.get("category"),
            is_active=_bool_arg("is_active"),
            search=request.args.get("search"),
        )
    except ServiceError as err:
        return _service_error(err)

    return (
        jsonify(
            {
                "items": products_response_schema.dump(result.items),
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


@products_bp.post("/products")
@role_required([UserRole.admin])
def create_product():
    """Create a catalog product (admin only). Duplicate name -> 409."""
    try:
        data = product_create_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        product = product_service.create_product(data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(product_response_schema.dump(product)), 201


@products_bp.get("/products/<int:product_id>")
@role_required(_ALL_ROLES)
def get_product(product_id: int):
    """Product detail including how many proposals reference it."""
    try:
        product = product_service.get_product(product_id)
    except ServiceError as err:
        return _service_error(err)

    return jsonify(product_detail_schema.dump(product)), 200


@products_bp.put("/products/<int:product_id>")
@role_required([UserRole.admin])
def update_product(product_id: int):
    """Partial update, including the ``is_active`` toggle (admin only)."""
    try:
        data = product_update_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        product = product_service.update_product(product_id, data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(product_response_schema.dump(product)), 200


@products_bp.delete("/products/<int:product_id>")
@role_required([UserRole.admin])
def delete_product(product_id: int):
    """Retire a product (admin only). Rows are never hard-deleted."""
    try:
        product_service.soft_delete_product(product_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify({"message": "Product deleted."}), 200
