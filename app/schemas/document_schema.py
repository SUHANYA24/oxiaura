"""Schemas for the document upload / OCR endpoints (BUILD_SPEC Phase 5).

The upload is ``multipart/form-data`` (the file rides alongside form fields), so
``DocumentUploadSchema`` validates only the non-file fields pulled from
``request.form``. Response schemas never expose the internal ``file_path``; the
stored SHA-256 is surfaced for integrity checks.
"""

from marshmallow import EXCLUDE, RAISE, Schema, fields, validate

from ..models import DocType, VerificationStatus


class DocumentUploadSchema(Schema):
    """Validates the form fields of ``POST /documents/upload``."""

    class Meta:
        unknown = RAISE

    customer_id = fields.Integer(required=True)
    doc_type = fields.Enum(DocType, by_value=True, required=True)


class DocumentResponseSchema(Schema):
    """Metadata for a stored document."""

    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(dump_only=True)
    customer_id = fields.Integer(dump_only=True)
    doc_type = fields.Enum(DocType, by_value=True, dump_only=True)
    sha256_hash = fields.String(dump_only=True)
    ocr_confidence = fields.Float(dump_only=True, allow_none=True)
    verification_status = fields.Enum(VerificationStatus, by_value=True, dump_only=True)
    uploaded_at = fields.DateTime(dump_only=True)


class OcrResultSchema(Schema):
    """Extracted fields + confidence for ``GET /documents/{id}/ocr-result``."""

    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(dump_only=True)
    ocr_confidence = fields.Float(dump_only=True, allow_none=True)
    extracted_fields = fields.Raw(dump_only=True, allow_none=True)
    verification_status = fields.Enum(VerificationStatus, by_value=True, dump_only=True)


class DocumentDetailSchema(DocumentResponseSchema):
    """Metadata plus the OCR payload — returned by upload and detail views."""

    extracted_fields = fields.Raw(dump_only=True, allow_none=True)


class FraudReportSchema(Schema):
    """Fraud sub-scores + weighted aggregate + verdict for a document."""

    class Meta:
        unknown = EXCLUDE

    document_id = fields.Integer(dump_only=True)
    ela_score = fields.Float(dump_only=True, allow_none=True)
    cnn_fraud_score = fields.Float(dump_only=True, allow_none=True)
    siamese_similarity = fields.Float(dump_only=True, allow_none=True)
    aggregate_score = fields.Float(dump_only=True, allow_none=True)
    is_flagged = fields.Boolean(dump_only=True)
    flag_reason = fields.String(dump_only=True, allow_none=True)
    checked_at = fields.DateTime(dump_only=True)
    verdict = fields.Function(
        lambda log: "flagged" if log.is_flagged else "clear", dump_only=True
    )


class DocumentVerifySchema(Schema):
    """Validates ``POST /documents/{id}/verify`` (admin approve/reject)."""

    class Meta:
        unknown = RAISE

    decision = fields.String(
        required=True, validate=validate.OneOf(["verified", "rejected"])
    )
    reason = fields.String(
        required=False, allow_none=True, validate=validate.Length(max=250)
    )


# Singletons for reuse in routes.
document_upload_schema = DocumentUploadSchema()
document_response_schema = DocumentResponseSchema()
document_detail_schema = DocumentDetailSchema()
ocr_result_schema = OcrResultSchema()
fraud_report_schema = FraudReportSchema()
document_verify_schema = DocumentVerifySchema()
