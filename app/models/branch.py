"""Branch model — a physical office/location of the business."""

from datetime import datetime

from ..extensions import db


class Branch(db.Model):
    __tablename__ = "branches"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    # ``use_alter`` breaks the circular FK dependency with ``users`` (users
    # also references branches). The constraint is emitted as a separate
    # ALTER TABLE after both tables exist so migrations apply cleanly.
    manager_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "users.id", use_alter=True, name="fk_branches_manager_id_users"
        ),
        nullable=True,
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # The user who manages this branch (Branch.manager_id → users.id).
    manager = db.relationship(
        "User",
        foreign_keys=[manager_id],
        back_populates="managed_branches",
    )
    # All users that belong to this branch (User.branch_id → branches.id).
    users = db.relationship(
        "User",
        foreign_keys="User.branch_id",
        back_populates="branch",
    )
    

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Branch {self.id} {self.name!r}>"
