"""Shared pytest fixtures for the Phase 3 auth test suite.

Everything runs against the testing config (in-memory SQLite), so the suite
needs no MySQL or Redis. A fresh schema and user set are created per test for
isolation, and the JWT blocklist is cleared between tests.
"""

import pytest

from app import create_app
from app.extensions import db as _db
from app.models import User, UserRole
from app.utils.security import hash_password
from app.utils import token_blocklist

# Credentials reused across tests.
ADMIN_EMAIL = "admin@test.local"
ADMIN_PASSWORD = "Admin@123"
REP_EMAIL = "rep@test.local"
REP_PASSWORD = "Rep@1234"
REP2_EMAIL = "rep2@test.local"
REP2_PASSWORD = "Rep2@1234"
STAFF_EMAIL = "staff@test.local"
STAFF_PASSWORD = "Staff@123"
DISABLED_EMAIL = "disabled@test.local"
DISABLED_PASSWORD = "Disabled@123"


@pytest.fixture
def app():
    """A fresh application + schema for each test."""
    app = create_app("testing")
    # A JWT secret comfortably above the HMAC-SHA256 minimum key length.
    app.config["JWT_SECRET_KEY"] = "test-secret-key-of-sufficient-length-1234567890"

    with app.app_context():
        _db.create_all()
        _seed_users()
        yield app
        _db.session.remove()
        _db.drop_all()

    # Ensure no revoked tokens leak into the next test.
    token_blocklist.clear()


def _seed_users():
    users = [
        User(
            full_name="Admin User",
            email=ADMIN_EMAIL,
            password_hash=hash_password(ADMIN_PASSWORD),
            role=UserRole.admin,
            is_active=True,
        ),
        User(
            full_name="Sales Rep",
            email=REP_EMAIL,
            password_hash=hash_password(REP_PASSWORD),
            role=UserRole.sales_rep,
            is_active=True,
        ),
        User(
            full_name="Second Sales Rep",
            email=REP2_EMAIL,
            password_hash=hash_password(REP2_PASSWORD),
            role=UserRole.sales_rep,
            is_active=True,
        ),
        User(
            full_name="Head Office Staff",
            email=STAFF_EMAIL,
            password_hash=hash_password(STAFF_PASSWORD),
            role=UserRole.head_office_staff,
            is_active=True,
        ),
        User(
            full_name="Disabled User",
            email=DISABLED_EMAIL,
            password_hash=hash_password(DISABLED_PASSWORD),
            role=UserRole.sales_rep,
            is_active=False,
        ),
    ]
    _db.session.add_all(users)
    _db.session.commit()


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, email, password):
    return client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )


def _auth_header(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def login():
    """Helper: ``login(client, email, password)`` -> response."""
    return _login


@pytest.fixture
def auth_header():
    """Helper: ``auth_header(token)`` -> Authorization header dict."""
    return _auth_header


@pytest.fixture
def admin_tokens(client):
    """Return (access_token, refresh_token) for the seeded admin."""
    resp = _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    body = resp.get_json()
    return body["access_token"], body["refresh_token"]


@pytest.fixture
def rep_tokens(client):
    resp = _login(client, REP_EMAIL, REP_PASSWORD)
    body = resp.get_json()
    return body["access_token"], body["refresh_token"]


@pytest.fixture
def token_for(client):
    """Helper: ``token_for(email, password)`` -> access token string."""

    def _token(email, password):
        return _login(client, email, password).get_json()["access_token"]

    return _token


@pytest.fixture
def user_id_by_email(app):
    """Helper: ``user_id_by_email(email)`` -> the seeded user's id."""

    def _lookup(email):
        return User.query.filter_by(email=email).one().id

    return _lookup
