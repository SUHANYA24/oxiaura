"""Product service — business logic for the admin-managed product catalog.

Access rules:
  * Every mutation is ``admin``-only, enforced at the route with
    ``role_required``. There is no row-level scoping here — the catalog is
    org-wide reference data, so any authenticated role may read it (a sales rep
    has to be able to pick a product when submitting a proposal).
  * Products are never hard-deleted: ``proposals.product_id`` references this
    table, so ``DELETE`` sets ``is_deleted`` and historical proposals keep
    resolving.

All database access goes through the ORM — no raw SQL (BUILD_SPEC section 3,
rule 6). State changes are logged with the acting user id (rule 9).
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal

from sqlalchemy import func, or_, select

from ..extensions import db
from ..models import Product, ProductCategory, User
from .errors import ConflictError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)

# Product codes look like "PRD-1041". New codes start here and increment.
_CODE_PREFIX = "PRD-"
_CODE_START = 1001
_CODE_RE = re.compile(r"^PRD-(\d+)$")

_MAX_PER_PAGE = 100
_DEFAULT_PER_PAGE = 20


def _generate_product_code() -> str:
    """Return the next sequential product code (e.g. ``PRD-1042``).

    Derived from the highest existing numeric suffix so codes don't collide after
    deletes. Soft-deleted rows are included so a code is never reused.
    """
    codes = db.session.query(Product.product_code).all()
    highest = _CODE_START - 1
    for (code,) in codes:
        match = _CODE_RE.match(code or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{_CODE_PREFIX}{highest + 1}"


def _name_exists(name: str, *, exclude_id: int | None = None) -> bool:
    """True if a live (non-deleted) product already uses this name.

    Case-insensitive: preventing "Teak Unit" / "teak unit" from coexisting is the
    whole point of replacing the old free-text ``product_type``. A soft-deleted
    product does not reserve its name — that row is retired.
    """
    query = Product.query.filter(
        func.lower(Product.name) == name.strip().lower(),
        Product.is_deleted.is_(False),
    )
    if exclude_id is not None:
        query = query.filter(Product.id != exclude_id)
    return db.session.query(query.exists()).scalar()


def _check_amount_window(minimum, maximum) -> None:
    """Reject a max below the min. Both values may be ``None``."""
    if minimum is not None and maximum is not None and Decimal(maximum) < Decimal(minimum):
        raise ValidationError(
            "max_investment must be greater than or equal to min_investment."
        )


def create_product(data: dict, current_user: User) -> Product:
    """Create a catalog product. Rejects a duplicate name and auto-assigns a code."""
    name = data["name"].strip()
    if _name_exists(name):
        raise ConflictError(f"A product named '{name}' already exists.")

    _check_amount_window(data["min_investment"], data.get("max_investment"))

    product = Product(
        product_code=_generate_product_code(),
        name=name,
        category=data["category"],
        description=data.get("description"),
        min_investment=data["min_investment"],
        max_investment=data.get("max_investment"),
        duration_months=data["duration_months"],
        interest_rate=data["interest_rate"],
        is_active=data.get("is_active", True),
    )
    db.session.add(product)
    db.session.commit()
    logger.info(
        "Product %s (%s) created by user %s",
        product.id,
        product.product_code,
        current_user.id,
    )
    return product


def list_products(
    *,
    page: int = 1,
    per_page: int = _DEFAULT_PER_PAGE,
    category: str | None = None,
    is_active: bool | None = None,
    search: str | None = None,
):
    """Return a paginated, filtered list of live products."""
    page = max(page, 1)
    per_page = min(max(per_page, 1), _MAX_PER_PAGE)

    stmt = select(Product).where(Product.is_deleted.is_(False))

    if category is not None:
        try:
            category_enum = ProductCategory(category)
        except ValueError:
            raise ValidationError(
                f"Invalid category '{category}'. Expected one of: "
                f"{', '.join(c.value for c in ProductCategory)}."
            )
        stmt = stmt.where(Product.category == category_enum)

    if is_active is not None:
        stmt = stmt.where(Product.is_active.is_(is_active))

    if search:
        term = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Product.name.ilike(term),
                Product.product_code.ilike(term),
                Product.description.ilike(term),
            )
        )

    stmt = stmt.order_by(Product.id.desc())
    return db.paginate(stmt, page=page, per_page=per_page, error_out=False)


def get_product(product_id: int) -> Product:
    """Fetch a single live product, or raise :class:`NotFoundError`."""
    product = Product.query.filter(
        Product.id == product_id, Product.is_deleted.is_(False)
    ).first()
    if product is None:
        raise NotFoundError("Product not found.")
    return product


def resolve_selectable_product(product_id: int) -> Product:
    """Return a product that may be attached to a proposal.

    Stricter than :func:`get_product`: a retired (``is_deleted``) or off-sale
    (``is_active`` false) product is rejected as a ``ValidationError`` so the
    caller surfaces 422 on the ``product_id`` field rather than a bare 404 that
    reads as "the proposal doesn't exist".
    """
    product = Product.query.filter(
        Product.id == product_id, Product.is_deleted.is_(False)
    ).first()
    if product is None:
        raise ValidationError(f"product_id {product_id} does not reference a product.")
    if not product.is_active:
        raise ValidationError(
            f"Product '{product.name}' is not currently available for new proposals."
        )
    return product


def update_product(product_id: int, data: dict, current_user: User) -> Product:
    """Apply a partial update. Re-checks name uniqueness and the amount window."""
    product = get_product(product_id)

    if "name" in data:
        name = data["name"].strip()
        if name.lower() != product.name.lower() and _name_exists(
            name, exclude_id=product.id
        ):
            raise ConflictError(f"A product named '{name}' already exists.")
        product.name = name

    # The window has to be validated against the merged result: a payload that
    # lowers only max_investment must still be checked against the stored min.
    minimum = data.get("min_investment", product.min_investment)
    maximum = (
        data.get("max_investment")
        if "max_investment" in data
        else product.max_investment
    )
    _check_amount_window(minimum, maximum)

    for field in (
        "category",
        "description",
        "min_investment",
        "max_investment",
        "duration_months",
        "interest_rate",
        "is_active",
    ):
        if field in data:
            setattr(product, field, data[field])

    db.session.commit()
    logger.info(
        "Product %s (%s) updated by user %s",
        product.id,
        product.product_code,
        current_user.id,
    )
    return product


def soft_delete_product(product_id: int, current_user: User) -> None:
    """Retire a product. Route restricts this to admins.

    The row is kept so that proposals referencing it still resolve; it simply
    stops appearing in the catalog and can no longer be selected.
    """
    product = get_product(product_id)
    product.is_deleted = True
    db.session.commit()
    logger.info(
        "Product %s (%s) soft-deleted by user %s",
        product.id,
        product.product_code,
        current_user.id,
    )
