"""Integration tests for the Phase 4 customer CRUD vertical slice.

Covers the acceptance criteria: full CRUD, duplicate-NIC rejection, pagination,
filters, and role scoping (sales_rep limited to their own customers, admin-only
soft delete).
"""

import pytest

from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    REP_EMAIL,
    REP_PASSWORD,
    REP2_EMAIL,
    REP2_PASSWORD,
    STAFF_EMAIL,
    STAFF_PASSWORD,
)

BASE = "/api/v1/customers"


@pytest.fixture
def admin_h(token_for, auth_header):
    return auth_header(token_for(ADMIN_EMAIL, ADMIN_PASSWORD))


@pytest.fixture
def rep_h(token_for, auth_header):
    return auth_header(token_for(REP_EMAIL, REP_PASSWORD))


@pytest.fixture
def rep2_h(token_for, auth_header):
    return auth_header(token_for(REP2_EMAIL, REP2_PASSWORD))


@pytest.fixture
def staff_h(token_for, auth_header):
    return auth_header(token_for(STAFF_EMAIL, STAFF_PASSWORD))


def _make(client, headers, **overrides):
    payload = {"nic_number": "199012345678", "full_name": "Nimal Perera"}
    payload.update(overrides)
    return client.post(BASE, headers=headers, json=payload)


class TestCreate:
    def test_rep_creates_customer_assigned_to_self(
        self, client, rep_h, user_id_by_email
    ):
        resp = _make(client, rep_h, nic_number="900000000001")
        assert resp.status_code == 201
        body = resp.get_json()
        assert body["assigned_rep_id"] == user_id_by_email(REP_EMAIL)
        assert body["status"] == "pending"
        assert body["customer_code"].startswith("C-")

    def test_customer_code_is_auto_assigned_and_sequential(self, client, admin_h):
        first = _make(client, admin_h, nic_number="900000000010").get_json()
        second = _make(client, admin_h, nic_number="900000000011").get_json()
        n1 = int(first["customer_code"].split("-")[1])
        n2 = int(second["customer_code"].split("-")[1])
        assert n2 == n1 + 1

    def test_rep_cannot_assign_to_another_rep(
        self, client, rep_h, user_id_by_email
    ):
        # assigned_rep_id is ignored for a sales_rep; it forces self-assignment.
        other = user_id_by_email(REP2_EMAIL)
        body = _make(client, rep_h, nic_number="900000000002", assigned_rep_id=other).get_json()
        assert body["assigned_rep_id"] == user_id_by_email(REP_EMAIL)

    def test_admin_can_assign_to_a_specific_rep(
        self, client, admin_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP2_EMAIL)
        body = _make(client, admin_h, nic_number="900000000003", assigned_rep_id=rep_id).get_json()
        assert body["assigned_rep_id"] == rep_id

    def test_admin_assign_to_nonexistent_rep_returns_422(self, client, admin_h):
        resp = _make(client, admin_h, nic_number="900000000004", assigned_rep_id=99999)
        assert resp.status_code == 422

    def test_duplicate_nic_rejected(self, client, admin_h):
        _make(client, admin_h, nic_number="123456789V")
        dup = _make(client, admin_h, nic_number="123456789V", full_name="Someone Else")
        assert dup.status_code == 409
        assert dup.get_json()["error"] == "conflict"

    def test_missing_required_field_returns_422(self, client, admin_h):
        resp = client.post(BASE, headers=admin_h, json={"full_name": "No NIC"})
        assert resp.status_code == 422
        assert "nic_number" in resp.get_json()["messages"]

    def test_unknown_field_rejected(self, client, admin_h):
        resp = client.post(
            BASE,
            headers=admin_h,
            json={"nic_number": "5", "full_name": "X", "hacker": True},
        )
        assert resp.status_code == 422
        assert "hacker" in resp.get_json()["messages"]

    def test_create_requires_authentication(self, client):
        resp = _make(client, {})
        assert resp.status_code == 401


class TestListPaginationAndFilters:
    def test_admin_sees_all_customers(self, client, admin_h, rep_h):
        _make(client, rep_h, nic_number="900000000101")
        _make(client, admin_h, nic_number="900000000102")
        resp = client.get(BASE, headers=admin_h)
        assert resp.status_code == 200
        assert resp.get_json()["pagination"]["total"] == 2

    def test_pagination_limits_and_reports_metadata(self, client, admin_h):
        for i in range(5):
            _make(client, admin_h, nic_number=f"90000020000{i}")
        resp = client.get(f"{BASE}?page=1&per_page=2", headers=admin_h)
        body = resp.get_json()
        assert len(body["items"]) == 2
        assert body["pagination"]["total"] == 5
        assert body["pagination"]["pages"] == 3
        assert body["pagination"]["has_next"] is True
        assert body["pagination"]["has_prev"] is False

    def test_per_page_is_capped(self, client, admin_h):
        resp = client.get(f"{BASE}?per_page=9999", headers=admin_h)
        assert resp.get_json()["pagination"]["per_page"] == 100

    def test_filter_by_status(self, client, admin_h):
        c = _make(client, admin_h, nic_number="900000000301").get_json()
        client.put(f"{BASE}/{c['id']}", headers=admin_h, json={"status": "verified"})
        _make(client, admin_h, nic_number="900000000302")  # stays pending
        verified = client.get(f"{BASE}?status=verified", headers=admin_h).get_json()
        assert verified["pagination"]["total"] == 1
        assert verified["items"][0]["status"] == "verified"

    def test_filter_by_invalid_status_returns_422(self, client, admin_h):
        resp = client.get(f"{BASE}?status=bogus", headers=admin_h)
        assert resp.status_code == 422

    def test_filter_by_assigned_rep(self, client, admin_h, user_id_by_email):
        rep2 = user_id_by_email(REP2_EMAIL)
        _make(client, admin_h, nic_number="900000000401", assigned_rep_id=rep2)
        _make(client, admin_h, nic_number="900000000402")  # assigned to admin
        resp = client.get(f"{BASE}?assigned_rep={rep2}", headers=admin_h)
        body = resp.get_json()
        assert body["pagination"]["total"] == 1
        assert body["items"][0]["assigned_rep_id"] == rep2

    def test_search_matches_name_nic_or_code(self, client, admin_h):
        _make(client, admin_h, nic_number="900000000501", full_name="Kamala Silva")
        _make(client, admin_h, nic_number="900000000502", full_name="Sunil Fernando")
        by_name = client.get(f"{BASE}?search=Kamala", headers=admin_h).get_json()
        assert by_name["pagination"]["total"] == 1
        by_nic = client.get(f"{BASE}?search=900000000502", headers=admin_h).get_json()
        assert by_nic["pagination"]["total"] == 1


class TestRoleScoping:
    def test_rep_lists_only_their_own(self, client, admin_h, rep_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        _make(client, rep_h, nic_number="900000000601")  # rep's own
        _make(client, admin_h, nic_number="900000000602")  # admin's own
        resp = client.get(BASE, headers=rep_h)
        body = resp.get_json()
        assert body["pagination"]["total"] == 1
        assert all(item["assigned_rep_id"] == rep_id for item in body["items"])

    def test_rep_cannot_read_another_reps_customer(self, client, rep_h, rep2_h):
        created = _make(client, rep_h, nic_number="900000000701").get_json()
        resp = client.get(f"{BASE}/{created['id']}", headers=rep2_h)
        assert resp.status_code == 403

    def test_rep_cannot_update_another_reps_customer(self, client, rep_h, rep2_h):
        created = _make(client, rep_h, nic_number="900000000702").get_json()
        resp = client.put(
            f"{BASE}/{created['id']}", headers=rep2_h, json={"full_name": "Hijack"}
        )
        assert resp.status_code == 403

    def test_head_office_staff_has_full_read_access(self, client, staff_h, rep_h):
        created = _make(client, rep_h, nic_number="900000000703").get_json()
        resp = client.get(f"{BASE}/{created['id']}", headers=staff_h)
        assert resp.status_code == 200

    def test_rep_cannot_reassign_via_update(self, client, rep_h, user_id_by_email):
        created = _make(client, rep_h, nic_number="900000000704").get_json()
        resp = client.put(
            f"{BASE}/{created['id']}",
            headers=rep_h,
            json={"assigned_rep_id": user_id_by_email(REP2_EMAIL)},
        )
        assert resp.status_code == 403


class TestGetDetail:
    def test_detail_includes_rep_documents_and_proposals(self, client, admin_h):
        created = _make(client, admin_h, nic_number="900000000801").get_json()
        resp = client.get(f"{BASE}/{created['id']}", headers=admin_h)
        body = resp.get_json()
        assert resp.status_code == 200
        assert body["assigned_rep"]["id"] == created["assigned_rep_id"]
        assert body["documents"] == []
        assert body["proposals"] == []

    def test_detail_of_missing_customer_returns_404(self, client, admin_h):
        assert client.get(f"{BASE}/999999", headers=admin_h).status_code == 404


class TestUpdate:
    def test_update_fields(self, client, admin_h):
        created = _make(client, admin_h, nic_number="900000000901").get_json()
        resp = client.put(
            f"{BASE}/{created['id']}",
            headers=admin_h,
            json={"full_name": "Updated Name", "phone": "0770000000"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["full_name"] == "Updated Name"
        assert body["phone"] == "0770000000"

    def test_update_to_duplicate_nic_rejected(self, client, admin_h):
        _make(client, admin_h, nic_number="900000001001")
        other = _make(client, admin_h, nic_number="900000001002").get_json()
        resp = client.put(
            f"{BASE}/{other['id']}", headers=admin_h, json={"nic_number": "900000001001"}
        )
        assert resp.status_code == 409

    def test_update_unknown_field_rejected(self, client, admin_h):
        created = _make(client, admin_h, nic_number="900000001003").get_json()
        resp = client.put(
            f"{BASE}/{created['id']}", headers=admin_h, json={"nope": 1}
        )
        assert resp.status_code == 422


class TestSoftDelete:
    def test_admin_soft_deletes_and_customer_disappears(self, client, admin_h):
        created = _make(client, admin_h, nic_number="900000001101").get_json()
        assert client.delete(f"{BASE}/{created['id']}", headers=admin_h).status_code == 200
        # Hidden from detail and list afterwards.
        assert client.get(f"{BASE}/{created['id']}", headers=admin_h).status_code == 404
        listed = client.get(BASE, headers=admin_h).get_json()
        assert all(item["id"] != created["id"] for item in listed["items"])

    def test_deleted_nic_can_be_reused(self, client, admin_h):
        created = _make(client, admin_h, nic_number="900000001201").get_json()
        client.delete(f"{BASE}/{created['id']}", headers=admin_h)
        # NIC now free because dedup ignores soft-deleted rows.
        resp = _make(client, admin_h, nic_number="900000001201")
        assert resp.status_code == 201

    def test_sales_rep_cannot_delete(self, client, rep_h):
        created = _make(client, rep_h, nic_number="900000001301").get_json()
        assert client.delete(f"{BASE}/{created['id']}", headers=rep_h).status_code == 403

    def test_head_office_staff_cannot_delete(self, client, staff_h, admin_h):
        created = _make(client, admin_h, nic_number="900000001302").get_json()
        assert client.delete(f"{BASE}/{created['id']}", headers=staff_h).status_code == 403

    def test_delete_missing_customer_returns_404(self, client, admin_h):
        assert client.delete(f"{BASE}/999999", headers=admin_h).status_code == 404
