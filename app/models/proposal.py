"""Proposal model — multi-stage investment proposal workflow."""

import enum
from datetime import datetime

from ..extensions import db


class ProposalWorkflowStatus(str, enum.Enum):
    submitted = "submitted"
    rep_review = "rep_review"
    ho_review = "ho_review"
    approved = "approved"
    rejected = "rejected"


class Proposal(db.Model):
    __tablename__ = "proposals"

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(
        db.Integer, db.ForeignKey("customers.id"), nullable=False
    )
    sales_rep_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False
    )
    proposed_amount = db.Column(db.Numeric(14, 2), nullable=False)
    product_type = db.Column(db.String(100), nullable=True)
    workflow_status = db.Column(
        db.Enum(ProposalWorkflowStatus),
        default=ProposalWorkflowStatus.submitted,
        nullable=False,
    )
    notes = db.Column(db.Text, nullable=True)
    submitted_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )

    customer = db.relationship("Customer", back_populates="proposals")
    sales_rep = db.relationship("User", back_populates="proposals")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Proposal {self.id} {self.workflow_status.value}>"
