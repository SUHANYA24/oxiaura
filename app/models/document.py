"""Document model — uploaded customer documents (NIC, bank slip, etc.)."""

import enum
from datetime import datetime

from ..extensions import db


class DocType(str, enum.Enum):
    nic = "nic"
    bank_slip = "bank_slip"
    bank_book = "bank_book"
    proposal_form = "proposal_form"


class VerificationStatus(str, enum.Enum):
    pending = "pending"
    verified = "verified"
    rejected = "rejected"


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(
        db.Integer, db.ForeignKey("customers.id"), nullable=False
    )
    doc_type = db.Column(db.Enum(DocType), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    sha256_hash = db.Column(db.String(64), index=True, nullable=False)
    ocr_confidence = db.Column(db.Float, nullable=True)
    extracted_fields = db.Column(db.JSON, nullable=True)
    verification_status = db.Column(
        db.Enum(VerificationStatus),
        default=VerificationStatus.pending,
        nullable=False,
    )
    uploaded_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )

    customer = db.relationship("Customer", back_populates="documents")
    # One-to-one with FraudLog.
    fraud_log = db.relationship(
        "FraudLog",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Document {self.id} {self.doc_type.value}>"
