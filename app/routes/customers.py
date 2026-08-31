"""Customer CRUD routes.

Flow per request: role guard -> Marshmallow schema -> service -> ORM -> schema
response. Role scoping (sales_rep sees only their own customers) is enforced in
the service; the decorators here gate which roles may reach each endpoint.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.customer_schema import (
    customer_create_schema,
    customer_detail_schema,
    customer_response_schema,
    customer_update_schema,
    customers_response_schema,
)
from ..services import customer_service
from ..services.errors import ServiceError
from ..utils.security import role_required

customers_bp = Blueprint("customers", __name__)

_ALL_ROLES = [UserRole.admin, UserRole.head_office_staff, UserRole.sales_rep]


def _current_user() -> User:
    """Load the authenticated user from the JWT identity."""
    return db.session.get(User, int(get_jwt_identity()))


def _service_error(err: ServiceError):
    return jsonify({"error": err.error, "message": err.message}), err.status


def _validation_error(err: SchemaValidationError):
    return jsonify({"error": "validation_error", "messages": err.messages}), 422


def _int_arg(name: str, default: int) -> int:
    """Parse an integer query arg, falling back to ``default`` if absent/invalid."""
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@customers_bp.get("/customers")
@role_required(_ALL_ROLES)
def list_customers():
    """Paginated, filterable list. Sales reps are scoped to their own."""
    try:
        result = customer_service.list_customers(
            _current_user(),
            page=_int_arg("page", 1),
            per_page=_int_arg("per_page", 20),
            status=request.args.get("status"),
            assigned_rep=_int_arg("assigned_rep", None),
            search=request.args.get("search"),
        )
    except ServiceError as err:
        return _service_error(err)

    return (
        jsonify(
            {
                "items": customers_response_schema.dump(result.items),
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


@customers_bp.post("/customers")
@role_required(_ALL_ROLES)
def create_customer():
    """Register a new customer (auto customer_code, NIC dedup)."""
    try:
        data = customer_create_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        customer = customer_service.create_customer(data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(customer_response_schema.dump(customer)), 201


@customers_bp.get("/customers/<int:customer_id>")
@role_required(_ALL_ROLES)
def get_customer(customer_id: int):
    """Customer detail including assigned rep, documents, and proposals."""
    try:
        customer = customer_service.get_customer(customer_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(customer_detail_schema.dump(customer)), 200


@customers_bp.put("/customers/<int:customer_id>")
@role_required(_ALL_ROLES)
def update_customer(customer_id: int):
    """Partial update. Sales reps may update only their own customers."""
    try:
        data = customer_update_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        customer = customer_service.update_customer(customer_id, data, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(customer_response_schema.dump(customer)), 200


@customers_bp.delete("/customers/<int:customer_id>")
@role_required([UserRole.admin])
def delete_customer(customer_id: int):
    """Soft delete a customer (admin only)."""
    try:
        customer_service.soft_delete_customer(customer_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify({"message": "Customer deleted."}), 200
