"""OCR service — orchestrates upload, storage, and field extraction.

Flow (BUILD_SPEC Phase 5): validate customer access -> validate + store the
file (UUID name, SHA-256) -> create the ``documents`` row -> preprocess ->
OCR -> parse fields -> persist ``extracted_fields`` + ``ocr_confidence``.

Phase 7 splits this into two halves: :func:`create_document` does the fast,
request-thread work (access check, storage, DB row) and :func:`process_document`
runs the heavy OCR + fraud pipeline, invoked by a Celery worker off the request
thread. Business logic only — no Flask HTTP types.
"""

from __future__ import annotations

import logging

from ..ai.ocr import extractor, preprocessor
from ..extensions import db
from ..models import Customer, DocType, Document, User
from ..utils import file_handler
from . import customer_service, fraud_service
from .errors import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


def run_ocr_pipeline(file_path: str) -> dict:
    """Preprocess the image, run OCR, and parse fields.

    :returns: the structured payload from :func:`extractor.parse_fields`.
    """
    image = preprocessor.preprocess(file_path)
    lines = extractor.run_ocr(image)
    return extractor.parse_fields(lines)


def _apply_ocr(document: Document) -> None:
    """Run the pipeline for ``document`` and persist the results in place.

    A failure never discards the stored file: the document keeps a diagnostic
    payload and a null confidence so it can be re-processed or reviewed.
    """
    try:
        result = run_ocr_pipeline(document.file_path)
        document.extracted_fields = result
        document.ocr_confidence = result.get("mean_confidence")
    except Exception as exc:  # noqa: BLE001 - keep the upload; record the failure
        logger.exception("OCR failed for document %s", document.id)
        document.extracted_fields = {"error": "ocr_failed", "detail": str(exc)}
        document.ocr_confidence = None


def _apply_fraud(document: Document) -> None:
    """Run the fraud pipeline for ``document`` (does not commit).

    Guarded like OCR: a detector failure must not discard the upload.
    """
    try:
        fraud_service.analyze_document(document)
    except Exception:  # noqa: BLE001 - keep the upload; record nothing on failure
        logger.exception("Fraud analysis failed for document %s", document.id)


def upload_document(
    customer_id: int, doc_type_value: str, file_storage, current_user: User
) -> Document:
    """Deprecated synchronous upload (pre-Phase 7).

    Retained for callers/tests that still want the whole pipeline inline. The
    HTTP upload route now uses :func:`create_document` + a Celery task instead.
    """
    document = create_document(customer_id, doc_type_value, file_storage, current_user)
    return process_document(document.id)


def create_document(
    customer_id: int, doc_type_value: str, file_storage, current_user: User
) -> Document:
    """Validate + store an uploaded document, without running OCR/fraud.

    This is the synchronous half of the Phase 7 flow: it does only the fast,
    request-thread work (access check, file validation, storage, DB row) and
    commits so the document has an id. The heavy OCR + fraud pipeline is run
    afterwards by :func:`process_document` on a Celery worker.

    :raises NotFoundError / ForbiddenError: propagated from the customer scope
        check (a sales_rep may only upload for their own customers).
    :raises ValidationError / UnsupportedMediaError: from file validation.
    """
    try:
        doc_type = DocType(doc_type_value)
    except ValueError:
        raise ValidationError(
            f"Invalid doc_type '{doc_type_value}'. Expected one of: "
            f"{', '.join(d.value for d in DocType)}."
        )

    # Enforces existence + row-level access (sales_rep scoped to own customers).
    customer: Customer = customer_service.get_customer(customer_id, current_user)

    data, mime = file_handler.validate_upload(file_storage)
    file_path, sha256_hash = file_handler.store_file(data, mime)

    document = Document(
        customer_id=customer.id,
        doc_type=doc_type,
        file_path=file_path,
        sha256_hash=sha256_hash,
    )
    db.session.add(document)
    db.session.commit()
    return document


def process_document(document_id: int) -> Document | None:
    """Run OCR + fraud for a stored document and persist the results.

    Called from the Celery task (no HTTP context, no access check — the row was
    already access-checked at upload time). Returns the updated document, or
    ``None`` if it no longer exists.
    """
    document = db.session.get(Document, document_id)
    if document is None:
        logger.warning("process_document: document %s not found", document_id)
        return None

    _apply_ocr(document)
    db.session.flush()  # ensure document.id is available for fraud/notifications
    _apply_fraud(document)
    db.session.commit()
    return document


def get_document(document_id: int, current_user: User) -> Document:
    """Fetch a document, enforcing access via its customer's scope."""
    document = db.session.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found.")
    # Reuse the customer access rules; raises Forbidden/NotFound as appropriate.
    customer_service.get_customer(document.customer_id, current_user)
    return document


def reprocess_document(document_id: int, current_user: User) -> Document:
    """Re-run OCR on an existing document (e.g. after a failure)."""
    document = get_document(document_id, current_user)
    _apply_ocr(document)
    db.session.commit()
    return document
