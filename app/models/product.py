"""Product model — the prebuilt catalog of plantation investment products.

Admin-managed reference data. A ``Proposal`` points at a row here instead of
carrying a free-text product name, so reporting can group by a stable id.

Two independent off-switches, both admin-only:
  * ``is_active`` — temporarily off-sale, reversible. Still visible in the
    catalog, but cannot be attached to a new proposal.
  * ``is_deleted`` — retired. Hidden from every read. Never a hard delete,
    because ``proposals.product_id`` references this table and historical
    proposals must keep resolving.
"""

import enum
from datetime import datetime

from ..extensions import db


class ProductCategory(str, enum.Enum):
    teak = "teak"
    agarwood = "agarwood"
    coconut = "coconut"
    mixed = "mixed"
    other = "other"


class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    product_code = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(150), nullable=False, index=True)
    category = db.Column(
        db.Enum(ProductCategory),
        default=ProductCategory.other,
        nullable=False,
    )
    description = db.Column(db.Text, nullable=True)
    min_investment = db.Column(db.Numeric(14, 2), nullable=False)
    max_investment = db.Column(db.Numeric(14, 2), nullable=True)
    duration_months = db.Column(db.Integer, nullable=False)
    # Percent per annum, e.g. 12.5 -> 12.5% p.a.
    interest_rate = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False, index=True)
    is_deleted = db.Column(
        db.Boolean, default=False, nullable=False, index=True
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    proposals = db.relationship("Proposal", back_populates="product")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Product {self.id} {self.product_code!r}>"
