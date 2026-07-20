"""Agreement model — signed investment agreements with QR authenticity."""

import enum

from ..extensions import db


class AgreementStatus(str, enum.Enum):
    pending = "pending"
    active = "active"
    cancelled = "cancelled"


class Agreement(db.Model):
    __tablename__ = "agreements"

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(
        db.Integer, db.ForeignKey("customers.id"), nullable=False
    )
    created_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False
    )
    agreement_number = db.Column(db.String(30), unique=True, nullable=False)
    investment_amount = db.Column(db.Numeric(14, 2), nullable=False)
    duration_months = db.Column(db.Integer, nullable=False)
    interest_rate = db.Column(db.Float, nullable=False)
    qr_code_token = db.Column(db.String(255), unique=True, nullable=True)
    pdf_path = db.Column(db.String(255), nullable=True)
    status = db.Column(
        db.Enum(AgreementStatus),
        default=AgreementStatus.pending,
        nullable=False,
    )
    signed_at = db.Column(db.DateTime, nullable=True)

    customer = db.relationship("Customer", back_populates="agreements")
    creator = db.relationship("User", back_populates="agreements")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Agreement {self.id} {self.agreement_number!r}>"
