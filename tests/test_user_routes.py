"""Integration tests for the user administration module.

Covers CRUD over operator accounts, duplicate-email rejection, pagination and
filters, password reset (admin) vs. change (self-service), and the guards that
keep an admin from locking themselves — or the system — out of the admin role.
"""

import pytest

from app.extensions import db as _db
from app.models import Branch, User, UserRole
from app.services import user_service
from app.services.errors import ConflictError
from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    DISABLED_EMAIL,
    REP_EMAIL,
    REP_PASSWORD,
    STAFF_EMAIL,
    STAFF_PASSWORD,
)

BASE = "/api/v1/users"

# Satisfies the schema policy: 8-72 chars with at least one letter and digit.
NEW_PASSWORD = "Renewed@2026"


@pytest.fixture
def admin_h(token_for, auth_header):
    return auth_header(token_for(ADMIN_EMAIL, ADMIN_PASSWORD))


@pytest.fixture
def rep_h(token_for, auth_header):
    return auth_header(token_for(REP_EMAIL, REP_PASSWORD))


@pytest.fixture
def staff_h(token_for, auth_header):
    return auth_header(token_for(STAFF_EMAIL, STAFF_PASSWORD))


@pytest.fixture
def branch_id(app):
    """A persisted branch, for exercising the ``branch_id`` reference."""
    branch = Branch(name="Colombo HQ", location="Colombo 03")
    _db.session.add(branch)
    _db.session.commit()
    return branch.id


def _make(client, headers, **overrides):
    payload = {
        "full_name": "Kamal Silva",
        "email": "kamal@test.local",
        "password": "Kamal@1234",
        "role": "sales_rep",
    }
    payload.update(overrides)
    return client.post(BASE, headers=headers, json=payload)


class TestCreate:
    def test_admin_creates_user(self, client, admin_h):
        resp = _make(client, admin_h)
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["email"] == "kamal@test.local"
        assert body["role"] == "sales_rep"
        assert body["is_active"] is True
        assert body["id"] > 0

    def test_response_never_exposes_the_password(self, client, admin_h):
        body = _make(client, admin_h).get_json()
        assert "password" not in body
        assert "password_hash" not in body

    def test_created_user_can_log_in(self, client, admin_h, login):
        _make(client, admin_h, email="loginme@test.local", password="Login@1234")
        resp = login(client, "loginme@test.local", "Login@1234")
        assert resp.status_code == 200
        assert resp.get_json()["user"]["email"] == "loginme@test.local"

    def test_email_is_normalized_to_lowercase(self, client, admin_h):
        body = _make(client, admin_h, email="MixedCase@Test.Local").get_json()
        assert body["email"] == "mixedcase@test.local"

    def test_can_create_an_inactive_account(self, client, admin_h):
        body = _make(client, admin_h, is_active=False).get_json()
        assert body["is_active"] is False

    def test_branch_can_be_assigned(self, client, admin_h, branch_id):
        body = _make(client, admin_h, branch_id=branch_id).get_json()
        assert body["branch_id"] == branch_id

    def test_duplicate_email_rejected(self, client, admin_h):
        _make(client, admin_h)
        dup = _make(client, admin_h, full_name="Someone Else")
        assert dup.status_code == 409
        assert dup.get_json()["error"] == "conflict"

    def test_duplicate_email_differing_only_in_case_rejected(self, client, admin_h):
        _make(client, admin_h, email="dupe@test.local")
        dup = _make(client, admin_h, email="DUPE@test.local")
        assert dup.status_code == 409

    def test_email_matching_a_seeded_user_rejected(self, client, admin_h):
        assert _make(client, admin_h, email=REP_EMAIL).status_code == 409

    def test_nonexistent_branch_returns_422(self, client, admin_h):
        resp = _make(client, admin_h, branch_id=999999)
        assert resp.status_code == 422
        assert resp.get_json()["error"] == "validation_error"

    def test_missing_required_field_returns_422(self, client, admin_h):
        payload = {"email": "x@test.local", "password": "Xxxx@1234", "role": "sales_rep"}
        resp = client.post(BASE, headers=admin_h, json=payload)
        assert resp.status_code == 422
        assert "full_name" in resp.get_json()["messages"]

    def test_unknown_field_rejected(self, client, admin_h):
        resp = _make(client, admin_h, is_superuser=True)
        assert resp.status_code == 422
        assert "is_superuser" in resp.get_json()["messages"]

    def test_invalid_role_returns_422(self, client, admin_h):
        resp = _make(client, admin_h, role="superadmin")
        assert resp.status_code == 422
        assert "role" in resp.get_json()["messages"]

    def test_short_password_returns_422(self, client, admin_h):
        resp = _make(client, admin_h, password="Ab@1")
        assert resp.status_code == 422
        assert "password" in resp.get_json()["messages"]

    def test_password_without_a_digit_returns_422(self, client, admin_h):
        resp = _make(client, admin_h, password="NoDigitsHere")
        assert resp.status_code == 422
        assert "password" in resp.get_json()["messages"]

    def test_password_without_a_letter_returns_422(self, client, admin_h):
        resp = _make(client, admin_h, password="12345678")
        assert resp.status_code == 422
        assert "password" in resp.get_json()["messages"]

    def test_create_requires_authentication(self, client):
        assert _make(client, {}).status_code == 401

    def test_head_office_staff_cannot_create(self, client, staff_h):
        assert _make(client, staff_h).status_code == 403

    def test_sales_rep_cannot_create(self, client, rep_h):
        assert _make(client, rep_h).status_code == 403


class TestListPaginationAndFilters:
    def test_lists_every_seeded_user(self, client, admin_h):
        body = client.get(BASE, headers=admin_h).get_json()
        assert body["pagination"]["total"] == 5

    def test_pagination_limits_and_reports_metadata(self, client, admin_h):
        body = client.get(f"{BASE}?page=1&per_page=2", headers=admin_h).get_json()
        assert len(body["items"]) == 2
        pagination = body["pagination"]
        assert pagination["page"] == 1
        assert pagination["per_page"] == 2
        assert pagination["total"] == 5
        assert pagination["pages"] == 3
        assert pagination["has_next"] is True
        assert pagination["has_prev"] is False

    def test_per_page_is_capped(self, client, admin_h):
        body = client.get(f"{BASE}?per_page=5000", headers=admin_h).get_json()
        assert body["pagination"]["per_page"] == 100

    def test_invalid_per_page_falls_back_to_the_default(self, client, admin_h):
        body = client.get(f"{BASE}?per_page=abc", headers=admin_h).get_json()
        assert body["pagination"]["per_page"] == 20

    def test_filter_by_role(self, client, admin_h):
        body = client.get(f"{BASE}?role=sales_rep", headers=admin_h).get_json()
        # Two active reps plus the disabled one, which is also a sales_rep.
        assert body["pagination"]["total"] == 3
        assert all(item["role"] == "sales_rep" for item in body["items"])

    def test_filter_by_is_active(self, client, admin_h):
        active = client.get(f"{BASE}?is_active=true", headers=admin_h).get_json()
        assert active["pagination"]["total"] == 4
        inactive = client.get(f"{BASE}?is_active=false", headers=admin_h).get_json()
        assert inactive["pagination"]["total"] == 1
        assert inactive["items"][0]["email"] == DISABLED_EMAIL

    def test_filter_by_branch(self, client, admin_h, branch_id):
        _make(client, admin_h, email="branched@test.local", branch_id=branch_id)
        body = client.get(f"{BASE}?branch_id={branch_id}", headers=admin_h).get_json()
        assert body["pagination"]["total"] == 1
        assert body["items"][0]["email"] == "branched@test.local"

    def test_search_matches_full_name(self, client, admin_h):
        body = client.get(f"{BASE}?search=Head Office", headers=admin_h).get_json()
        assert body["pagination"]["total"] == 1
        assert body["items"][0]["email"] == STAFF_EMAIL

    def test_search_matches_email(self, client, admin_h):
        body = client.get(f"{BASE}?search=rep2@", headers=admin_h).get_json()
        assert body["pagination"]["total"] == 1

    def test_invalid_role_filter_returns_422(self, client, admin_h):
        resp = client.get(f"{BASE}?role=wizard", headers=admin_h)
        assert resp.status_code == 422
        assert "sales_rep" in resp.get_json()["message"]

    def test_head_office_staff_may_list(self, client, staff_h):
        assert client.get(BASE, headers=staff_h).status_code == 200

    def test_sales_rep_cannot_list(self, client, rep_h):
        assert client.get(BASE, headers=rep_h).status_code == 403

    def test_list_requires_authentication(self, client):
        assert client.get(BASE).status_code == 401


class TestGetDetail:
    def test_detail_includes_the_branch(self, client, admin_h, branch_id, user_id_by_email):
        client.put(
            f"{BASE}/{user_id_by_email(REP_EMAIL)}",
            headers=admin_h,
            json={"branch_id": branch_id},
        )
        body = client.get(
            f"{BASE}/{user_id_by_email(REP_EMAIL)}", headers=admin_h
        ).get_json()
        assert body["branch"]["name"] == "Colombo HQ"
        assert body["branch_id"] == branch_id

    def test_detail_of_missing_user_returns_404(self, client, admin_h):
        resp = client.get(f"{BASE}/999999", headers=admin_h)
        assert resp.status_code == 404
        assert resp.get_json()["error"] == "not_found"

    def test_head_office_staff_may_read(self, client, staff_h, user_id_by_email):
        resp = client.get(f"{BASE}/{user_id_by_email(REP_EMAIL)}", headers=staff_h)
        assert resp.status_code == 200

    def test_sales_rep_cannot_read(self, client, rep_h, user_id_by_email):
        resp = client.get(f"{BASE}/{user_id_by_email(REP_EMAIL)}", headers=rep_h)
        assert resp.status_code == 403


class TestUpdate:
    def test_admin_updates_name_and_email(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(
            f"{BASE}/{rep_id}",
            headers=admin_h,
            json={"full_name": "Renamed Rep", "email": "renamed@test.local"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["full_name"] == "Renamed Rep"
        assert body["email"] == "renamed@test.local"

    def test_admin_changes_another_users_role(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        body = client.put(
            f"{BASE}/{rep_id}", headers=admin_h, json={"role": "head_office_staff"}
        ).get_json()
        assert body["role"] == "head_office_staff"

    def test_branch_can_be_cleared_with_null(
        self, client, admin_h, branch_id, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        client.put(f"{BASE}/{rep_id}", headers=admin_h, json={"branch_id": branch_id})
        body = client.put(
            f"{BASE}/{rep_id}", headers=admin_h, json={"branch_id": None}
        ).get_json()
        assert body["branch_id"] is None

    def test_duplicate_email_rejected(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(f"{BASE}/{rep_id}", headers=admin_h, json={"email": STAFF_EMAIL})
        assert resp.status_code == 409

    def test_unchanged_email_is_not_a_conflict(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(f"{BASE}/{rep_id}", headers=admin_h, json={"email": REP_EMAIL})
        assert resp.status_code == 200

    def test_nonexistent_branch_returns_422(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(f"{BASE}/{rep_id}", headers=admin_h, json={"branch_id": 999999})
        assert resp.status_code == 422

    def test_password_cannot_be_changed_through_update(
        self, client, admin_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(
            f"{BASE}/{rep_id}", headers=admin_h, json={"password": NEW_PASSWORD}
        )
        assert resp.status_code == 422
        assert "password" in resp.get_json()["messages"]

    def test_update_of_missing_user_returns_404(self, client, admin_h):
        resp = client.put(f"{BASE}/999999", headers=admin_h, json={"full_name": "Ghost"})
        assert resp.status_code == 404

    def test_admin_cannot_change_own_role(self, client, admin_h, user_id_by_email):
        admin_id = user_id_by_email(ADMIN_EMAIL)
        resp = client.put(f"{BASE}/{admin_id}", headers=admin_h, json={"role": "sales_rep"})
        assert resp.status_code == 403
        assert resp.get_json()["error"] == "forbidden"

    def test_admin_cannot_deactivate_self_through_update(
        self, client, admin_h, user_id_by_email
    ):
        admin_id = user_id_by_email(ADMIN_EMAIL)
        resp = client.put(f"{BASE}/{admin_id}", headers=admin_h, json={"is_active": False})
        assert resp.status_code == 403

    def test_admin_may_edit_own_profile_fields(self, client, admin_h, user_id_by_email):
        admin_id = user_id_by_email(ADMIN_EMAIL)
        resp = client.put(
            f"{BASE}/{admin_id}", headers=admin_h, json={"full_name": "Chief Admin"}
        )
        assert resp.status_code == 200
        assert resp.get_json()["full_name"] == "Chief Admin"

    def test_restating_own_role_unchanged_is_allowed(
        self, client, admin_h, user_id_by_email
    ):
        # No actual change, so the self-protection guard must not trip.
        admin_id = user_id_by_email(ADMIN_EMAIL)
        resp = client.put(f"{BASE}/{admin_id}", headers=admin_h, json={"role": "admin"})
        assert resp.status_code == 200

    def test_head_office_staff_cannot_update(self, client, staff_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(f"{BASE}/{rep_id}", headers=staff_h, json={"full_name": "Nope"})
        assert resp.status_code == 403

    def test_sales_rep_cannot_update(self, client, rep_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.put(f"{BASE}/{rep_id}", headers=rep_h, json={"full_name": "Nope"})
        assert resp.status_code == 403


class TestDeactivate:
    def test_admin_deactivates_a_user(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.delete(f"{BASE}/{rep_id}", headers=admin_h)
        assert resp.status_code == 200
        assert resp.get_json()["user"]["is_active"] is False

    def test_deactivated_user_cannot_log_in(
        self, client, admin_h, login, user_id_by_email
    ):
        client.delete(f"{BASE}/{user_id_by_email(REP_EMAIL)}", headers=admin_h)
        resp = login(client, REP_EMAIL, REP_PASSWORD)
        assert resp.status_code == 403

    def test_the_row_is_retained_rather_than_deleted(
        self, client, admin_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        client.delete(f"{BASE}/{rep_id}", headers=admin_h)
        assert _db.session.get(User, rep_id) is not None

    def test_deactivating_twice_is_idempotent(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        client.delete(f"{BASE}/{rep_id}", headers=admin_h)
        again = client.delete(f"{BASE}/{rep_id}", headers=admin_h)
        assert again.status_code == 200
        assert again.get_json()["user"]["is_active"] is False

    def test_admin_cannot_deactivate_self(self, client, admin_h, user_id_by_email):
        admin_id = user_id_by_email(ADMIN_EMAIL)
        resp = client.delete(f"{BASE}/{admin_id}", headers=admin_h)
        assert resp.status_code == 403
        assert resp.get_json()["error"] == "forbidden"

    def test_deactivate_of_missing_user_returns_404(self, client, admin_h):
        assert client.delete(f"{BASE}/999999", headers=admin_h).status_code == 404

    def test_head_office_staff_cannot_deactivate(
        self, client, staff_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        assert client.delete(f"{BASE}/{rep_id}", headers=staff_h).status_code == 403

    def test_sales_rep_cannot_deactivate(self, client, rep_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        assert client.delete(f"{BASE}/{rep_id}", headers=rep_h).status_code == 403


class TestLastAdminGuard:
    """The last-admin guard is defence in depth.

    It cannot be reached over HTTP: the mutating routes are admin-only, so any
    caller other than the target is itself a second active admin, and a caller
    acting on themselves is stopped earlier by the self-protection guard. These
    tests drive the service directly to prove the guard holds if the route gate
    is ever loosened.
    """

    def test_service_refuses_to_deactivate_the_only_active_admin(self, app):
        admin = User.query.filter_by(email=ADMIN_EMAIL).one()
        staff = User.query.filter_by(email=STAFF_EMAIL).one()
        with pytest.raises(ConflictError):
            user_service.deactivate_user(admin.id, staff)

    def test_service_refuses_to_demote_the_only_active_admin(self, app):
        admin = User.query.filter_by(email=ADMIN_EMAIL).one()
        staff = User.query.filter_by(email=STAFF_EMAIL).one()
        with pytest.raises(ConflictError):
            user_service.update_user(
                admin.id, {"role": UserRole.sales_rep}, staff
            )

    def test_demoting_an_admin_is_fine_while_another_remains(self, app):
        staff = User.query.filter_by(email=STAFF_EMAIL).one()
        spare = User(
            full_name="Spare Admin",
            email="spare@test.local",
            password_hash="x",
            role=UserRole.admin,
            is_active=True,
        )
        _db.session.add(spare)
        _db.session.commit()

        admin = User.query.filter_by(email=ADMIN_EMAIL).one()
        updated = user_service.update_user(
            admin.id, {"role": UserRole.sales_rep}, staff
        )
        assert updated.role == UserRole.sales_rep


class TestPasswordReset:
    def test_admin_resets_another_users_password(
        self, client, admin_h, login, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{BASE}/{rep_id}/reset-password",
            headers=admin_h,
            json={"new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 200
        assert login(client, REP_EMAIL, NEW_PASSWORD).status_code == 200
        assert login(client, REP_EMAIL, REP_PASSWORD).status_code == 401

    def test_weak_password_returns_422(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{BASE}/{rep_id}/reset-password", headers=admin_h, json={"new_password": "abc"}
        )
        assert resp.status_code == 422
        assert "new_password" in resp.get_json()["messages"]

    def test_missing_body_returns_422(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(f"{BASE}/{rep_id}/reset-password", headers=admin_h, json={})
        assert resp.status_code == 422

    def test_reset_for_missing_user_returns_404(self, client, admin_h):
        resp = client.post(
            f"{BASE}/999999/reset-password",
            headers=admin_h,
            json={"new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 404

    def test_head_office_staff_cannot_reset(self, client, staff_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{BASE}/{rep_id}/reset-password",
            headers=staff_h,
            json={"new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 403

    def test_sales_rep_cannot_reset_another_users_password(
        self, client, rep_h, user_id_by_email
    ):
        staff_id = user_id_by_email(STAFF_EMAIL)
        resp = client.post(
            f"{BASE}/{staff_id}/reset-password",
            headers=rep_h,
            json={"new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 403


class TestChangeOwnPassword:
    URL = f"{BASE}/me/password"

    def test_sales_rep_changes_own_password(self, client, rep_h, login):
        resp = client.put(
            self.URL,
            headers=rep_h,
            json={"current_password": REP_PASSWORD, "new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 200
        assert login(client, REP_EMAIL, NEW_PASSWORD).status_code == 200
        assert login(client, REP_EMAIL, REP_PASSWORD).status_code == 401

    def test_admin_changes_own_password(self, client, admin_h, login):
        resp = client.put(
            self.URL,
            headers=admin_h,
            json={"current_password": ADMIN_PASSWORD, "new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 200
        assert login(client, ADMIN_EMAIL, NEW_PASSWORD).status_code == 200

    def test_wrong_current_password_returns_403(self, client, rep_h):
        resp = client.put(
            self.URL,
            headers=rep_h,
            json={"current_password": "WrongPass1", "new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 403
        assert resp.get_json()["error"] == "forbidden"

    def test_reusing_the_current_password_returns_422(self, client, rep_h):
        resp = client.put(
            self.URL,
            headers=rep_h,
            json={"current_password": REP_PASSWORD, "new_password": REP_PASSWORD},
        )
        assert resp.status_code == 422
        assert resp.get_json()["error"] == "validation_error"

    def test_weak_new_password_returns_422(self, client, rep_h):
        resp = client.put(
            self.URL,
            headers=rep_h,
            json={"current_password": REP_PASSWORD, "new_password": "short1"},
        )
        assert resp.status_code == 422
        assert "new_password" in resp.get_json()["messages"]

    def test_missing_current_password_returns_422(self, client, rep_h):
        resp = client.put(self.URL, headers=rep_h, json={"new_password": NEW_PASSWORD})
        assert resp.status_code == 422
        assert "current_password" in resp.get_json()["messages"]

    def test_requires_authentication(self, client):
        resp = client.put(
            self.URL,
            json={"current_password": REP_PASSWORD, "new_password": NEW_PASSWORD},
        )
        assert resp.status_code == 401
