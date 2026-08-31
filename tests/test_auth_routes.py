"""Integration tests for the Phase 3 auth endpoints.

Covers the acceptance criteria: login issues tokens, protected routes reject
missing/invalid tokens (401), role guards reject the wrong role (403), and
refresh/logout/revocation behave correctly.
"""

import jwt as pyjwt

from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    DISABLED_EMAIL,
    DISABLED_PASSWORD,
    REP_EMAIL,
    REP_PASSWORD,
    STAFF_EMAIL,
    STAFF_PASSWORD,
)


class TestLogin:
    def test_login_success_returns_tokens_and_user(self, client, login):
        resp = login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
        assert resp.status_code == 200
        body = resp.get_json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["user"]["email"] == ADMIN_EMAIL
        assert body["user"]["role"] == "admin"

    def test_login_never_exposes_password_hash(self, client, login):
        body = login(client, ADMIN_EMAIL, ADMIN_PASSWORD).get_json()
        assert "password_hash" not in body["user"]

    def test_access_token_carries_role_and_user_id_claims(self, client, login):
        body = login(client, ADMIN_EMAIL, ADMIN_PASSWORD).get_json()
        claims = pyjwt.decode(
            body["access_token"], options={"verify_signature": False}
        )
        assert claims["role"] == "admin"
        assert claims["user_id"] == 1
        assert claims["sub"] == "1"  # identity is the stringified id

    def test_login_wrong_password_returns_401(self, client, login):
        resp = login(client, ADMIN_EMAIL, "wrong-password")
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "authentication_failed"

    def test_login_unknown_email_returns_401(self, client, login):
        resp = login(client, "ghost@test.local", "whatever")
        assert resp.status_code == 401

    def test_login_disabled_account_returns_403(self, client, login):
        resp = login(client, DISABLED_EMAIL, DISABLED_PASSWORD)
        assert resp.status_code == 403

    def test_login_missing_fields_returns_422(self, client):
        resp = client.post("/api/v1/auth/login", json={"email": ADMIN_EMAIL})
        assert resp.status_code == 422
        assert "password" in resp.get_json()["messages"]

    def test_login_rejects_unknown_fields(self, client):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "hax": 1},
        )
        assert resp.status_code == 422
        assert "hax" in resp.get_json()["messages"]

    def test_login_invalid_email_format_returns_422(self, client):
        resp = client.post(
            "/api/v1/auth/login", json={"email": "not-an-email", "password": "x"}
        )
        assert resp.status_code == 422

    def test_login_empty_body_returns_422(self, client):
        resp = client.post("/api/v1/auth/login", json={})
        assert resp.status_code == 422


class TestProtectedRoute:
    def test_me_without_token_returns_401(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "authorization_required"

    def test_me_with_garbage_token_returns_401(self, client, auth_header):
        resp = client.get("/api/v1/auth/me", headers=auth_header("garbage.token.here"))
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "invalid_token"

    def test_me_with_valid_token_returns_profile(self, client, auth_header, admin_tokens):
        access, _ = admin_tokens
        resp = client.get("/api/v1/auth/me", headers=auth_header(access))
        assert resp.status_code == 200
        assert resp.get_json()["email"] == ADMIN_EMAIL

    def test_refresh_token_rejected_on_access_protected_route(
        self, client, auth_header, admin_tokens
    ):
        _, refresh = admin_tokens
        # Presenting a refresh token where an access token is required -> 401.
        resp = client.get("/api/v1/auth/me", headers=auth_header(refresh))
        assert resp.status_code == 401


class TestRoleEnforcement:
    def test_admin_allowed_on_admin_only(self, client, auth_header, admin_tokens):
        access, _ = admin_tokens
        resp = client.get("/api/v1/auth/admin-only", headers=auth_header(access))
        assert resp.status_code == 200

    def test_sales_rep_forbidden_on_admin_only(self, client, auth_header, rep_tokens):
        access, _ = rep_tokens
        resp = client.get("/api/v1/auth/admin-only", headers=auth_header(access))
        assert resp.status_code == 403
        assert resp.get_json()["error"] == "forbidden"

    def test_head_office_staff_forbidden_on_admin_only(
        self, client, login, auth_header
    ):
        access = login(client, STAFF_EMAIL, STAFF_PASSWORD).get_json()["access_token"]
        resp = client.get("/api/v1/auth/admin-only", headers=auth_header(access))
        assert resp.status_code == 403

    def test_admin_only_without_token_returns_401_not_403(self, client):
        resp = client.get("/api/v1/auth/admin-only")
        assert resp.status_code == 401


class TestRefresh:
    def test_refresh_issues_new_access_token(self, client, auth_header, admin_tokens):
        _, refresh = admin_tokens
        resp = client.post("/api/v1/auth/refresh", headers=auth_header(refresh))
        assert resp.status_code == 200
        new_access = resp.get_json()["access_token"]
        # New access token is usable and preserves claims.
        claims = pyjwt.decode(new_access, options={"verify_signature": False})
        assert claims["role"] == "admin"
        assert claims["user_id"] == 1
        assert claims["type"] == "access"

    def test_refresh_with_access_token_rejected(self, client, auth_header, admin_tokens):
        access, _ = admin_tokens
        resp = client.post("/api/v1/auth/refresh", headers=auth_header(access))
        assert resp.status_code == 401

    def test_refresh_without_token_returns_401(self, client):
        resp = client.post("/api/v1/auth/refresh")
        assert resp.status_code == 401


class TestLogoutRevocation:
    def test_logout_revokes_access_token(self, client, auth_header, admin_tokens):
        access, _ = admin_tokens
        # Works before logout.
        assert client.get("/api/v1/auth/me", headers=auth_header(access)).status_code == 200
        # Logout.
        logout = client.post("/api/v1/auth/logout", headers=auth_header(access))
        assert logout.status_code == 200
        # Same token is now revoked.
        after = client.get("/api/v1/auth/me", headers=auth_header(access))
        assert after.status_code == 401
        assert after.get_json()["error"] == "token_revoked"

    def test_logout_revokes_refresh_token(self, client, auth_header, admin_tokens):
        _, refresh = admin_tokens
        logout = client.post("/api/v1/auth/logout", headers=auth_header(refresh))
        assert logout.status_code == 200
        # Revoked refresh token can no longer mint access tokens.
        resp = client.post("/api/v1/auth/refresh", headers=auth_header(refresh))
        assert resp.status_code == 401

    def test_logout_without_token_returns_401(self, client):
        resp = client.post("/api/v1/auth/logout")
        assert resp.status_code == 401

    def test_revoking_one_token_does_not_revoke_the_other(
        self, client, auth_header, admin_tokens
    ):
        access, refresh = admin_tokens
        client.post("/api/v1/auth/logout", headers=auth_header(access))
        # Refresh token was not revoked, so it still works.
        resp = client.post("/api/v1/auth/refresh", headers=auth_header(refresh))
        assert resp.status_code == 200
