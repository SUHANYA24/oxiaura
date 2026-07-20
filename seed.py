"""Database seed script.

Inserts a baseline dataset so the app is usable immediately after migration:
  * one admin user (and one sample sales rep so customers have a valid rep)
  * two branches
  * a few sample customers

Run with:  python seed.py   (or:  flask --app wsgi shell < seed.py)

Idempotent: re-running skips rows that already exist, so it is safe to run
repeatedly during development.
"""

import os

import bcrypt

from app import create_app
from app.extensions import db
from app.models import Branch, Customer, CustomerStatus, User, UserRole


def _hash_password(plain: str) -> str:
    """bcrypt hash (cost 12) — matches the Phase 3 security helper contract."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode(
        "utf-8"
    )


def _get_or_create_user(**kwargs) -> User:
    user = User.query.filter_by(email=kwargs["email"]).first()
    if user:
        return user
    user = User(**kwargs)
    db.session.add(user)
    db.session.flush()
    return user


def _get_or_create_branch(name: str, **kwargs) -> Branch:
    branch = Branch.query.filter_by(name=name).first()
    if branch:
        return branch
    branch = Branch(name=name, **kwargs)
    db.session.add(branch)
    db.session.flush()
    return branch


def _get_or_create_customer(customer_code: str, **kwargs) -> Customer:
    customer = Customer.query.filter_by(customer_code=customer_code).first()
    if customer:
        return customer
    customer = Customer(customer_code=customer_code, **kwargs)
    db.session.add(customer)
    db.session.flush()
    return customer


def seed() -> None:
    admin_password = os.environ.get("SEED_ADMIN_PASSWORD", "Admin@123")
    rep_password = os.environ.get("SEED_REP_PASSWORD", "Rep@1234")

    # --- Users ---
    admin = _get_or_create_user(
        full_name="System Administrator",
        email="admin@plantvest.local",
        password_hash=_hash_password(admin_password),
        role=UserRole.admin,
        is_active=True,
    )
    rep = _get_or_create_user(
        full_name="Sample Sales Rep",
        email="rep@plantvest.local",
        password_hash=_hash_password(rep_password),
        role=UserRole.sales_rep,
        is_active=True,
    )

    # --- Branches ---
    colombo = _get_or_create_branch(
        "Colombo HQ", location="Colombo 03", manager_id=admin.id
    )
    kandy = _get_or_create_branch(
        "Kandy Branch", location="Kandy", manager_id=None
    )

    # Attach the sales rep to a branch.
    if rep.branch_id is None:
        rep.branch_id = colombo.id

    # --- Sample customers (assigned to the sample rep) ---
    _get_or_create_customer(
        "C-1001",
        nic_number="199012345678",
        full_name="Nimal Perera",
        address="12 Galle Road, Colombo",
        phone="0771234567",
        email="nimal@example.com",
        assigned_rep_id=rep.id,
        status=CustomerStatus.pending,
    )
    _get_or_create_customer(
        "C-1002",
        nic_number="198523456789",
        full_name="Kamala Silva",
        address="45 Kandy Road, Kandy",
        phone="0719876543",
        email="kamala@example.com",
        assigned_rep_id=rep.id,
        status=CustomerStatus.verified,
    )
    _get_or_create_customer(
        "C-1003",
        nic_number="200034567890",
        full_name="Sunil Fernando",
        address="8 Beach Road, Galle",
        phone="0765554321",
        email=None,
        assigned_rep_id=rep.id,
        status=CustomerStatus.flagged,
    )

    db.session.commit()

    print("Seed complete:")
    print(f"  admin user : {admin.email} (password: {admin_password})")
    print(f"  sales rep  : {rep.email} (password: {rep_password})")
    print(f"  branches   : {Branch.query.count()}")
    print(f"  customers  : {Customer.query.count()}")


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        seed()
