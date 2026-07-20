"""FraudLog model — fraud-detection results for a document (one-to-one)."""

from datetime import datetime

from ..extensions import db


class FraudLog(db.Model):
    __tablename__ = "fraud_logs"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(
        db.Integer,
        db.ForeignKey("documents.id"),
        unique=True,
        nullable=False,
    )
    ela_score = db.Column(db.Float, nullable=True)  # 0–100
    cnn_fraud_score = db.Column(db.Float, nullable=True)  # 0–1
    siamese_similarity = db.Column(db.Float, nullable=True)  # 0–1
    aggregate_score = db.Column(db.Float, nullable=True)  # 0–100 weighted
    is_flagged = db.Column(db.Boolean, default=False, nullable=False)
    flag_reason = db.Column(db.String(250), nullable=True)
    checked_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    document = db.relationship("Document", back_populates="fraud_log")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<FraudLog {self.id} doc={self.document_id} flagged={self.is_flagged}>"
