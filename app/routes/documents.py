"""Document upload & OCR routes (BUILD_SPEC Phase 5).

Flow mirrors the customer slice: role guard -> Marshmallow schema -> service ->
ORM -> schema response. Uploads are ``multipart/form-data`` (form fields +
``file``). All authenticated roles may reach these endpoints; row-level access
(a sales_rep only touching their own customers' documents) is enforced in the
service via the customer scope check.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity
from marshmallow import ValidationError as SchemaValidationError

from ..extensions import db
from ..models import User, UserRole
from ..schemas.document_schema import (
    document_detail_schema,
    document_response_schema,
    document_upload_schema,
    document_verify_schema,
    fraud_report_schema,
    ocr_result_schema,
)
from ..services import fraud_service, ocr_service
from ..services.errors import ServiceError
from ..utils.security import role_required

documents_bp = Blueprint("documents", __name__)

_ALL_ROLES = [UserRole.admin, UserRole.head_office_staff, UserRole.sales_rep]


def _current_user() -> User:
    return db.session.get(User, int(get_jwt_identity()))


def _service_error(err: ServiceError):
    return jsonify({"error": err.error, "message": err.message}), err.status


def _validation_error(err: SchemaValidationError):
    return jsonify({"error": "validation_error", "messages": err.messages}), 422


@documents_bp.post("/documents/upload")
@role_required(_ALL_ROLES)
def upload_document():
    """Upload a document -> store it -> run OCR -> return metadata + fields."""
    try:
        data = document_upload_schema.load(request.form.to_dict())
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        document = ocr_service.upload_document(
            customer_id=data["customer_id"],
            doc_type_value=data["doc_type"].value,
            file_storage=request.files.get("file"),
            current_user=_current_user(),
        )
    except ServiceError as err:
        return _service_error(err)

    return jsonify(document_detail_schema.dump(document)), 201


@documents_bp.get("/documents/<int:document_id>")
@role_required(_ALL_ROLES)
def get_document(document_id: int):
    """Return document metadata + verification status."""
    try:
        document = ocr_service.get_document(document_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(document_response_schema.dump(document)), 200


@documents_bp.get("/documents/<int:document_id>/ocr-result")
@role_required(_ALL_ROLES)
def get_ocr_result(document_id: int):
    """Return the extracted fields and confidence for a document."""
    try:
        document = ocr_service.get_document(document_id, _current_user())
    except ServiceError as err:
        return _service_error(err)

    return jsonify(ocr_result_schema.dump(document)), 200


@documents_bp.get("/documents/<int:document_id>/fraud-report")
@role_required(_ALL_ROLES)
def get_fraud_report(document_id: int):
    """Return the three fraud sub-scores + aggregate + verdict for a document."""
    try:
        document = ocr_service.get_document(document_id, _current_user())
        report = fraud_service.get_or_create_report(document)
    except ServiceError as err:
        return _service_error(err)

    return jsonify(fraud_report_schema.dump(report)), 200


@documents_bp.post("/documents/<int:document_id>/verify")
@role_required([UserRole.admin])
def verify_document(document_id: int):
    """Admin approve/reject a document's verification status."""
    try:
        data = document_verify_schema.load(request.get_json(silent=True) or {})
    except SchemaValidationError as err:
        return _validation_error(err)

    try:
        document = ocr_service.get_document(document_id, _current_user())
        document = fraud_service.verify_document(
            document, data["decision"], data.get("reason"), _current_user()
        )
    except ServiceError as err:
        return _service_error(err)

    return jsonify(document_response_schema.dump(document)), 200
