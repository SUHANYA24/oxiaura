"""Customer model — plantation investment customers."""

import enum
from datetime import datetime

from ..extensions import db


class CustomerStatus(str, enum.Enum):
    pending = "pending"
    verified = "verified"
    flagged = "flagged"


class Customer(db.Model):
    __tablename__ = "customers"

    id = db.Column(db.Integer, primary_key=True)
    customer_code = db.Column(db.String(20), unique=True, nullable=False)
    nic_number = db.Column(db.String(20), index=True, nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    address = db.Column(db.String(250), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(150), nullable=True)
    assigned_rep_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False
    )
    status = db.Column(
        db.Enum(CustomerStatus),
        default=CustomerStatus.pending,
        nullable=False,
    )
    registered_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )

    assigned_rep = db.relationship(
        "User", back_populates="assigned_customers"
    )
    documents = db.relationship(
        "Document", back_populates="customer", cascade="all, delete-orphan"
    )
    agreements = db.relationship("Agreement", back_populates="customer")
    proposals = db.relationship("Proposal", back_populates="customer")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Customer {self.id} {self.customer_code!r}>"
