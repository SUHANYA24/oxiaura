"""EmployeeTarget model — monthly KPI targets and actuals per user."""

from ..extensions import db


class EmployeeTarget(db.Model):
    __tablename__ = "employee_targets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False
    )
    month = db.Column(db.Integer, nullable=False)  # 1–12
    year = db.Column(db.Integer, nullable=False)
    target_customers = db.Column(db.Integer, nullable=False, default=0)
    actual_customers = db.Column(db.Integer, nullable=False, default=0)
    target_revenue = db.Column(db.Numeric(14, 2), nullable=False, default=0)
    actual_revenue = db.Column(db.Numeric(14, 2), nullable=False, default=0)

    user = db.relationship("User", back_populates="targets")

    __table_args__ = (
        db.UniqueConstraint(
            "user_id", "month", "year", name="uq_target_user_month_year"
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<EmployeeTarget u={self.user_id} {self.year}-{self.month:02d}>"
