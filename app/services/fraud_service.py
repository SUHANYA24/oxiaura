"""Fraud detection service (BUILD_SPEC Phase 6).

Runs the three detectors on a document, computes the weighted aggregate,
persists a one-to-one ``FraudLog``, and notifies verification staff when a
document is flagged. Also handles the admin verify/reject decision.

Aggregate = 0.3·ELA + 0.4·CNN·100 + 0.3·Siamese·100  (all normalized to 0–100);
a document is flagged when the aggregate exceeds ``FRAUD_FLAG_THRESHOLD``.
Business logic only — no Flask HTTP types.
"""

from __future__ import annotations

import logging

from flask import current_app

from ..ai.fraud import cnn_classifier, ela_detector, siamese_detector
from ..extensions import db
from ..models import Document, FraudLog, User, VerificationStatus
from . import notification_service
from .errors import NotFoundError, ValidationError

logger = logging.getLogger(__name__)

_ELA_WEIGHT = 0.3
_CNN_WEIGHT = 0.4
_SIAMESE_WEIGHT = 0.3

# Per-signal thresholds used only to compose a human-readable flag reason.
_ELA_CONCERN = 55.0
_CNN_CONCERN = 0.5
_SIAMESE_CONCERN = 0.7


def run_detectors(file_path: str) -> dict:
    """Run all three detectors and compute the weighted aggregate + verdict."""
    ela_score, ela_details = ela_detector.score_image(file_path)
    cnn_score = cnn_classifier.predict_tamper_probability(file_path)
    siamese_score = siamese_detector.highest_similarity(file_path)

    aggregate = round(
        _ELA_WEIGHT * ela_score
        + _CNN_WEIGHT * cnn_score * 100
        + _SIAMESE_WEIGHT * siamese_score * 100,
        2,
    )
    threshold = current_app.config.get("FRAUD_FLAG_THRESHOLD", 60)
    is_flagged = aggregate > threshold

    reasons = []
    if ela_score >= _ELA_CONCERN:
        reasons.append(f"high ELA ({ela_score})")
    if cnn_score >= _CNN_CONCERN:
        reasons.append(f"CNN tamper probability {cnn_score}")
    if siamese_score >= _SIAMESE_CONCERN:
        reasons.append(f"matches a known template (similarity {siamese_score})")
    if is_flagged and not reasons:
        reasons.append(f"elevated aggregate score ({aggregate})")

    return {
        "ela_score": ela_score,
        "cnn_fraud_score": cnn_score,
        "siamese_similarity": siamese_score,
        "aggregate_score": aggregate,
        "is_flagged": is_flagged,
        "flag_reason": "; ".join(reasons) if reasons else None,
        "ela_details": ela_details,
    }


def analyze_document(document: Document, *, notify: bool = True) -> FraudLog:
    """Score ``document`` and upsert its ``FraudLog`` (does not commit).

    On a flag, verification staff are notified within the same transaction.
    """
    scores = run_detectors(document.file_path)

    log = document.fraud_log or FraudLog(document=document)
    log.ela_score = scores["ela_score"]
    log.cnn_fraud_score = scores["cnn_fraud_score"]
    log.siamese_similarity = scores["siamese_similarity"]
    log.aggregate_score = scores["aggregate_score"]
    log.is_flagged = scores["is_flagged"]
    log.flag_reason = scores["flag_reason"]
    db.session.add(log)

    if scores["is_flagged"] and notify:
        customer = document.customer
        code = customer.customer_code if customer else document.customer_id
        notification_service.notify_staff(
            title="Document flagged for fraud review",
            message=(
                f"Document {document.id} ({document.doc_type.value}) for customer "
                f"{code} scored {scores['aggregate_score']} — {scores['flag_reason']}."
            ),
        )
    return log


def get_or_create_report(document: Document) -> FraudLog:
    """Return the document's fraud report, running the analysis if absent."""
    if document.fraud_log is not None:
        return document.fraud_log
    log = analyze_document(document)
    db.session.commit()
    return log


def verify_document(
    document: Document, decision: str, reason: str | None, current_user: User
) -> Document:
    """Admin approve/reject a document; notifies the assigned rep (commits)."""
    try:
        status = VerificationStatus(decision)
    except ValueError:
        raise ValidationError(
            f"Invalid decision '{decision}'. Expected 'verified' or 'rejected'."
        )
    if status not in (VerificationStatus.verified, VerificationStatus.rejected):
        raise ValidationError("Decision must be 'verified' or 'rejected'.")

    document.verification_status = status

    customer = document.customer
    if customer and customer.assigned_rep_id:
        suffix = f" Reason: {reason}" if reason else ""
        notification_service.notify_user(
            customer.assigned_rep_id,
            title=f"Document {status.value}",
            message=(
                f"Document {document.id} for customer {customer.customer_code} was "
                f"{status.value} by {current_user.full_name}.{suffix}"
            ),
        )

    logger.info(
        "Document %s %s by user %s", document.id, status.value, current_user.id
    )
    db.session.commit()
    return document
