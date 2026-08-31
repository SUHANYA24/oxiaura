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
import os
import threading

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
# These are BUILD_SPEC's example weights, kept as-is. Worth knowing before you
# tune them: ml_training.evaluate_pipeline measures ELA at AUC 0.485 and the
# Siamese term at 0.474 (it is still on its mock — no reference bank), against
# the CNN's 0.830. So 60% of the weight above currently carries no signal and the
# aggregate separates the classes *worse* than the CNN alone would. Re-weighting
# is a policy change and tests assert exact aggregates; see ml_training/README §8.

# Per-signal thresholds used only to compose a human-readable flag reason.
_ELA_CONCERN = 55.0
_CNN_CONCERN = 0.5
# Measured against the trained embedder (ml_training/README.md §5): a true match
# scores ~0.98, two different documents of the same kind ~0.70, so the separating
# cut-off is around 0.9. The old 0.7 was picked against mock scores and sits on
# top of the different-document population.
_SIAMESE_CONCERN = 0.95

# Keyed by bank path rather than a bare global so two apps in one process (the
# test suite makes several) cannot inherit each other's bank.
_reference_banks: dict[str, object | None] = {}
_reference_bank_lock = threading.Lock()


def _reference_embeddings():
    """Return the known-forgery embedding bank, or ``None`` if there is none.

    Loaded once per path per process: this is a ``[N, 512]`` tensor read from
    disk, and ``run_detectors`` is on the upload path. A missing or unreadable
    bank is not fatal — ``highest_similarity`` falls back to its mock — so this
    logs and returns ``None`` rather than failing an upload over it.
    """
    path = current_app.config.get("FRAUD_REFERENCE_BANK")
    if not path:
        return None
    if path in _reference_banks:
        return _reference_banks[path]

    with _reference_bank_lock:
        if path not in _reference_banks:
            bank = None
            if os.path.isfile(path):
                try:
                    import torch

                    bank = torch.load(path, map_location="cpu")
                except Exception:  # noqa: BLE001 - a bad bank must not break uploads
                    logger.exception("Could not load fraud reference bank at %s", path)
                    bank = None
            else:
                logger.warning(
                    "FRAUD_REFERENCE_BANK points at %s, which does not exist; "
                    "the Siamese detector will use its mock score",
                    path,
                )
            _reference_banks[path] = bank
    return _reference_banks[path]


def run_detectors(file_path: str) -> dict:
    """Run all three detectors and compute the weighted aggregate + verdict."""
    ela_score, ela_details = ela_detector.score_image(file_path)
    cnn_score = cnn_classifier.predict_tamper_probability(file_path)
    siamese_score = siamese_detector.highest_similarity(
        file_path, _reference_embeddings()
    )

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
