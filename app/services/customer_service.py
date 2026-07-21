"""Customer service — business logic for the customer vertical slice.

Access rules (BUILD_SPEC Phase 4):
  * ``sales_rep`` sees and manages only customers assigned to them.
  * ``admin`` / ``head_office_staff`` have full access.
  * Soft delete is admin-only (enforced at the route with ``role_required``).

All database access goes through the ORM — no raw SQL (section 3, rule 6).
"""

import re

from sqlalchemy import or_, select

from ..extensions import db
from ..models import Customer, CustomerStatus, User, UserRole
from .errors import ConflictError, ForbiddenError, NotFoundError, ValidationError

# Customer codes look like "C-1041". New codes start here and increment.
_CODE_PREFIX = "C-"
_CODE_START = 1001
_CODE_RE = re.compile(r"^C-(\d+)$")

# Roles with unrestricted access to every customer.
_FULL_ACCESS_ROLES = {UserRole.admin, UserRole.head_office_staff}

_MAX_PER_PAGE = 100
_DEFAULT_PER_PAGE = 20


def _has_full_access(user: User) -> bool:
    return user.role in _FULL_ACCESS_ROLES


def _generate_customer_code() -> str:
    """Return the next sequential customer code (e.g. ``C-1042``).

    Derived from the highest existing numeric suffix so codes don't collide
    after deletes. Soft-deleted rows are included so a code is never reused.
    """
    codes = db.session.query(Customer.customer_code).all()
    highest = _CODE_START - 1
    for (code,) in codes:
        match = _CODE_RE.match(code or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{_CODE_PREFIX}{highest + 1}"


def _resolve_assigned_rep(data: dict, current_user: User) -> int:
    """Determine the assigned_rep_id for a new customer, enforcing role rules."""
    if not _has_full_access(current_user):
        # A sales rep can only ever create customers for themselves.
        return current_user.id

    rep_id = data.get("assigned_rep_id", current_user.id)
    rep = db.session.get(User, rep_id)
    if rep is None or not rep.is_active:
        raise ValidationError("assigned_rep_id does not reference an active user.")
    return rep_id


def _nic_exists(nic_number: str, *, exclude_id: int | None = None) -> bool:
    """True if an active (non-deleted) customer already has this NIC."""
    query = Customer.query.filter(
        Customer.nic_number == nic_number,
        Customer.is_deleted.is_(False),
    )
    if exclude_id is not None:
        query = query.filter(Customer.id != exclude_id)
    return db.session.query(query.exists()).scalar()


def create_customer(data: dict, current_user: User) -> Customer:
    """Create a customer. Rejects a duplicate NIC and auto-assigns a code."""
    nic_number = data["nic_number"]
    if _nic_exists(nic_number):
        raise ConflictError(f"A customer with NIC {nic_number} already exists.")

    assigned_rep_id = _resolve_assigned_rep(data, current_user)

    customer = Customer(
        customer_code=_generate_customer_code(),
        nic_number=nic_number,
        full_name=data["full_name"],
        address=data.get("address"),
        phone=data.get("phone"),
        email=data.get("email"),
        assigned_rep_id=assigned_rep_id,
        status=CustomerStatus.pending,
    )
    db.session.add(customer)
    db.session.commit()
    return customer


def list_customers(
    current_user: User,
    *,
    page: int = 1,
    per_page: int = _DEFAULT_PER_PAGE,
    status: str | None = None,
    assigned_rep: int | None = None,
    search: str | None = None,
):
    """Return a paginated, filtered list of customers for ``current_user``."""
    page = max(page, 1)
    per_page = min(max(per_page, 1), _MAX_PER_PAGE)

    stmt = select(Customer).where(Customer.is_deleted.is_(False))

    # Sales reps only ever see their own customers.
    if not _has_full_access(current_user):
        stmt = stmt.where(Customer.assigned_rep_id == current_user.id)

    if status is not None:
        try:
            status_enum = CustomerStatus(status)
        except ValueError:
            raise ValidationError(
                f"Invalid status '{status}'. Expected one of: "
                f"{', '.join(s.value for s in CustomerStatus)}."
            )
        stmt = stmt.where(Customer.status == status_enum)

    # A sales_rep is already scoped to themselves; the assigned_rep filter only
    # meaningfully applies to full-access roles.
    if assigned_rep is not None and _has_full_access(current_user):
        stmt = stmt.where(Customer.assigned_rep_id == assigned_rep)

    if search:
        term = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Customer.full_name.ilike(term),
                Customer.nic_number.ilike(term),
                Customer.customer_code.ilike(term),
            )
        )

    stmt = stmt.order_by(Customer.id.desc())
    return db.paginate(stmt, page=page, per_page=per_page, error_out=False)


def get_customer(customer_id: int, current_user: User) -> Customer:
    """Fetch a single non-deleted customer, enforcing role scope."""
    customer = Customer.query.filter(
        Customer.id == customer_id, Customer.is_deleted.is_(False)
    ).first()
    if customer is None:
        raise NotFoundError("Customer not found.")
    if not _has_full_access(current_user) and customer.assigned_rep_id != current_user.id:
        raise ForbiddenError("You do not have access to this customer.")
    return customer


def update_customer(customer_id: int, data: dict, current_user: User) -> Customer:
    """Apply a partial update. Re-checks NIC dedup and role constraints."""
    customer = get_customer(customer_id, current_user)

    if "nic_number" in data and data["nic_number"] != customer.nic_number:
        if _nic_exists(data["nic_number"], exclude_id=customer.id):
            raise ConflictError(
                f"A customer with NIC {data['nic_number']} already exists."
            )
        customer.nic_number = data["nic_number"]

    # Reassigning a customer to a different rep is a full-access privilege.
    if "assigned_rep_id" in data:
        if not _has_full_access(current_user):
            raise ForbiddenError("You are not permitted to reassign customers.")
        rep = db.session.get(User, data["assigned_rep_id"])
        if rep is None or not rep.is_active:
            raise ValidationError("assigned_rep_id does not reference an active user.")
        customer.assigned_rep_id = data["assigned_rep_id"]

    for field in ("full_name", "address", "phone", "email"):
        if field in data:
            setattr(customer, field, data[field])

    if "status" in data:
        customer.status = data["status"]

    db.session.commit()
    return customer


def soft_delete_customer(customer_id: int, current_user: User) -> None:
    """Mark a customer deleted. Route restricts this to admins."""
    customer = Customer.query.filter(
        Customer.id == customer_id, Customer.is_deleted.is_(False)
    ).first()
    if customer is None:
        raise NotFoundError("Customer not found.")
    customer.is_deleted = True
    db.session.commit()
