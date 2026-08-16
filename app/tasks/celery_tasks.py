"""Celery tasks (BUILD_SPEC Phase 7).

The upload route stores the file synchronously (so the client gets a document
id and integrity hash back at once) and then enqueues :func:`process_document`
to run the heavy OCR + fraud pipeline off the request thread. The task is a
thin wrapper over :func:`app.services.ocr_service.process_document`; the Flask
application context is provided by the ``FlaskTask`` base class configured in
``celery_init_app`` so the ORM and services work exactly as in a request.

The AI models are still the deterministic mocks from Phases 5-6 (real weights
arrive later), so the pipeline is fully exercisable without GPUs or trained
models — the async plumbing is what Phase 7 adds.
"""

from __future__ import annotations

import logging

from celery import shared_task

from ..services import ocr_service

logger = logging.getLogger(__name__)


@shared_task(
    name="documents.process_document",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def process_document(self, document_id: int) -> dict:
    """Run OCR then fraud analysis for a stored document.

    Returns a small JSON-serializable summary so the result is meaningful when
    polled via the job-status endpoint. OCR/fraud sub-failures are already
    swallowed inside the service (the document is never lost); an unexpected
    error here is retried a couple of times before the task is marked failed.
    """
    try:
        document = ocr_service.process_document(document_id)
    except Exception as exc:  # noqa: BLE001 - retry transient failures
        logger.exception("process_document task failed for %s", document_id)
        raise self.retry(exc=exc)

    if document is None:
        return {"document_id": document_id, "status": "not_found"}

    fraud_log = document.fraud_log
    return {
        "document_id": document.id,
        "status": "processed",
        "ocr_confidence": document.ocr_confidence,
        "is_flagged": bool(fraud_log.is_flagged) if fraud_log else None,
        "aggregate_score": fraud_log.aggregate_score if fraud_log else None,
    }
