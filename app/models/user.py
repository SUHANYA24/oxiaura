"""User model — system operators (admin / head office staff / sales rep)."""

import enum
from datetime import datetime

from ..extensions import db


class UserRole(str, enum.Enum):
    admin = "admin"
    head_office_staff = "head_office_staff"
    sales_rep = "sales_rep"


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(UserRole), nullable=False)
    branch_id = db.Column(
        db.Integer, db.ForeignKey("branches.id"), nullable=True
    )
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Branch this user belongs to (User.branch_id → branches.id).
    branch = db.relationship(
        "Branch",
        foreign_keys=[branch_id],
        back_populates="users",
    )
    # Branches this user manages (Branch.manager_id → users.id).
    managed_branches = db.relationship(
        "Branch",
        foreign_keys="Branch.manager_id",
        back_populates="manager",
    )

    # Customers assigned to this user (as sales rep).
    assigned_customers = db.relationship(
        "Customer", back_populates="assigned_rep"
    )
    # Agreements created by this user.
    agreements = db.relationship("Agreement", back_populates="creator")
    # Proposals submitted by this user (as sales rep).
    proposals = db.relationship("Proposal", back_populates="sales_rep")
    # Monthly KPI targets for this user.
    targets = db.relationship("EmployeeTarget", back_populates="user")
    # Notifications addressed to this user.
    notifications = db.relationship("Notification", back_populates="user")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<User {self.id} {self.email!r} ({self.role.value})>"
