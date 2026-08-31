"""Integration tests for the admin-managed product catalog.

Covers the access split that defines this module — every authenticated role may
read the catalog, only ``admin`` may write it — plus code generation, the
case-insensitive name uniqueness that replaces the old free-text product_type,
the amount-window rule, filtering, and soft delete keeping proposal FKs intact.

conftest seeds one product (``PRD-1001``), so created products start at
``PRD-1002`` and list counts include that baseline row.
"""

import pytest

from app.extensions import db
from app.models import Product
from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    REP_EMAIL,
    REP_PASSWORD,
    SEED_PRODUCT_CODE,
    SEED_PRODUCT_NAME,
    STAFF_EMAIL,
    STAFF_PASSWORD,
)

CUSTOMERS = "/api/v1/customers"
PRODUCTS = "/api/v1/products"
PROPOSALS = "/api/v1/proposals"


@pytest.fixture
def admin_h(token_for, auth_header):
    return auth_header(token_for(ADMIN_EMAIL, ADMIN_PASSWORD))


@pytest.fixture
def staff_h(token_for, auth_header):
    return auth_header(token_for(STAFF_EMAIL, STAFF_PASSWORD))


@pytest.fixture
def rep_h(token_for, auth_header):
    return auth_header(token_for(REP_EMAIL, REP_PASSWORD))


def _payload(**overrides):
    payload = {
        "name": "Agarwood Growth Unit",
        "category": "agarwood",
        "description": "20-year agarwood unit.",
        "min_investment": "100000.00",
        "max_investment": "1000000.00",
        "duration_months": 240,
        "interest_rate": 14.0,
    }
    payload.update(overrides)
    return payload


def _create(client, headers, **overrides):
    return client.post(PRODUCTS, headers=headers, json=_payload(**overrides))


class TestRoleGating:
    """Writes are admin-only; reads are open to every authenticated role."""

    def test_admin_can_create(self, client, admin_h):
        assert _create(client, admin_h).status_code == 201

    def test_rep_cannot_create(self, client, rep_h):
        assert _create(client, rep_h).status_code == 403

    def test_staff_cannot_create(self, client, staff_h):
        assert _create(client, staff_h).status_code == 403

    def test_rep_cannot_update(self, client, rep_h, seed_product_id):
        resp = client.put(
            f"{PRODUCTS}/{seed_product_id}", headers=rep_h, json={"interest_rate": 99.0}
        )
        assert resp.status_code == 403

    def test_staff_cannot_update(self, client, staff_h, seed_product_id):
        resp = client.put(
            f"{PRODUCTS}/{seed_product_id}", headers=staff_h, json={"is_active": False}
        )
        assert resp.status_code == 403

    def test_rep_cannot_delete(self, client, rep_h, seed_product_id):
        assert client.delete(f"{PRODUCTS}/{seed_product_id}", headers=rep_h).status_code == 403

    def test_staff_cannot_delete(self, client, staff_h, seed_product_id):
        assert (
            client.delete(f"{PRODUCTS}/{seed_product_id}", headers=staff_h).status_code
            == 403
        )

    @pytest.mark.parametrize("role_header", ["admin_h", "staff_h", "rep_h"])
    def test_every_role_can_read(self, client, request, role_header, seed_product_id):
        headers = request.getfixturevalue(role_header)
        assert client.get(PRODUCTS, headers=headers).status_code == 200
        assert client.get(f"{PRODUCTS}/{seed_product_id}", headers=headers).status_code == 200

    def test_reads_require_authentication(self, client):
        assert client.get(PRODUCTS).status_code == 401

    def test_writes_require_authentication(self, client):
        assert client.post(PRODUCTS, json=_payload()).status_code == 401


class TestCreate:
    def test_returns_the_created_product(self, client, admin_h):
        body = _create(client, admin_h).get_json()
        assert body["name"] == "Agarwood Growth Unit"
        assert body["category"] == "agarwood"
        assert body["min_investment"] == "100000.00"
        assert body["duration_months"] == 240
        assert body["interest_rate"] == 14.0
        # Defaults to on-sale.
        assert body["is_active"] is True

    def test_codes_are_sequential_after_the_seeded_row(self, client, admin_h):
        first = _create(client, admin_h, name="Product One").get_json()
        second = _create(client, admin_h, name="Product Two").get_json()
        assert first["product_code"] == "PRD-1002"
        assert second["product_code"] == "PRD-1003"

    def test_duplicate_name_is_409(self, client, admin_h):
        _create(client, admin_h, name="Coconut Unit")
        assert _create(client, admin_h, name="Coconut Unit").status_code == 409

    def test_duplicate_name_is_case_insensitive(self, client, admin_h):
        # The whole point of the catalog: no "Teak Unit" / "teak unit" drift.
        assert _create(client, admin_h, name=SEED_PRODUCT_NAME.upper()).status_code == 409

    def test_name_is_trimmed(self, client, admin_h):
        body = _create(client, admin_h, name="  Padded Name  ").get_json()
        assert body["name"] == "Padded Name"

    def test_max_below_min_is_422(self, client, admin_h):
        resp = _create(
            client, admin_h, min_investment="500000.00", max_investment="100000.00"
        )
        assert resp.status_code == 422

    def test_max_investment_is_optional(self, client, admin_h):
        payload = _payload(name="No Ceiling")
        del payload["max_investment"]
        resp = client.post(PRODUCTS, headers=admin_h, json=payload)
        assert resp.status_code == 201
        assert resp.get_json()["max_investment"] is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"min_investment": "0"},
            {"duration_months": 0},
            {"duration_months": -12},
            {"interest_rate": -1.0},
            {"category": "bogus"},
            {"name": ""},
        ],
    )
    def test_invalid_values_are_422(self, client, admin_h, overrides):
        assert _create(client, admin_h, **overrides).status_code == 422

    def test_missing_required_field_is_422(self, client, admin_h):
        assert client.post(PRODUCTS, headers=admin_h, json={"name": "x"}).status_code == 422

    def test_unknown_field_is_422(self, client, admin_h):
        assert _create(client, admin_h, hacker="x").status_code == 422

    def test_server_owned_fields_are_rejected(self, client, admin_h):
        # product_code and is_deleted are never client-supplied.
        assert _create(client, admin_h, product_code="PRD-9999").status_code == 422
        assert _create(client, admin_h, is_deleted=True).status_code == 422


class TestUpdate:
    def test_partial_update(self, client, admin_h, seed_product_id):
        body = client.put(
            f"{PRODUCTS}/{seed_product_id}",
            headers=admin_h,
            json={"interest_rate": 13.75, "description": "Revised terms."},
        ).get_json()
        assert body["interest_rate"] == 13.75
        assert body["description"] == "Revised terms."
        # Untouched fields survive.
        assert body["name"] == SEED_PRODUCT_NAME

    def test_is_active_toggle(self, client, admin_h, seed_product_id):
        off = client.put(
            f"{PRODUCTS}/{seed_product_id}", headers=admin_h, json={"is_active": False}
        ).get_json()
        assert off["is_active"] is False
        on = client.put(
            f"{PRODUCTS}/{seed_product_id}", headers=admin_h, json={"is_active": True}
        ).get_json()
        assert on["is_active"] is True

    def test_rename_onto_existing_name_is_409(self, client, admin_h):
        other = _create(client, admin_h, name="Other Unit").get_json()["id"]
        resp = client.put(
            f"{PRODUCTS}/{other}", headers=admin_h, json={"name": SEED_PRODUCT_NAME}
        )
        assert resp.status_code == 409

    def test_renaming_to_its_own_name_is_allowed(self, client, admin_h, seed_product_id):
        resp = client.put(
            f"{PRODUCTS}/{seed_product_id}", headers=admin_h, json={"name": SEED_PRODUCT_NAME}
        )
        assert resp.status_code == 200

    def test_lowering_max_below_stored_min_is_422(self, client, admin_h, seed_product_id):
        # Seeded min is 50000; a max of 100 must be rejected against the stored min.
        resp = client.put(
            f"{PRODUCTS}/{seed_product_id}", headers=admin_h, json={"max_investment": "100.00"}
        )
        assert resp.status_code == 422

    def test_raising_min_above_stored_max_is_422(self, client, admin_h, seed_product_id):
        # Seeded max is 500000.
        resp = client.put(
            f"{PRODUCTS}/{seed_product_id}",
            headers=admin_h,
            json={"min_investment": "900000.00"},
        )
        assert resp.status_code == 422

    def test_unknown_field_is_422(self, client, admin_h, seed_product_id):
        resp = client.put(f"{PRODUCTS}/{seed_product_id}", headers=admin_h, json={"x": 1})
        assert resp.status_code == 422

    def test_missing_product_is_404(self, client, admin_h):
        assert client.put(f"{PRODUCTS}/999999", headers=admin_h, json={}).status_code == 404


class TestSoftDelete:
    def test_delete_hides_the_product(self, client, admin_h, seed_product_id):
        assert client.delete(f"{PRODUCTS}/{seed_product_id}", headers=admin_h).status_code == 200
        assert client.get(f"{PRODUCTS}/{seed_product_id}", headers=admin_h).status_code == 404
        assert client.get(PRODUCTS, headers=admin_h).get_json()["items"] == []

    def test_row_is_retained(self, client, app, admin_h, seed_product_id):
        client.delete(f"{PRODUCTS}/{seed_product_id}", headers=admin_h)
        with app.app_context():
            product = db.session.get(Product, seed_product_id)
        assert product is not None
        assert product.is_deleted is True

    def test_deleting_twice_is_404(self, client, admin_h, seed_product_id):
        client.delete(f"{PRODUCTS}/{seed_product_id}", headers=admin_h)
        assert client.delete(f"{PRODUCTS}/{seed_product_id}", headers=admin_h).status_code == 404

    def test_existing_proposal_still_resolves(self, client, admin_h, rep_h, seed_product_id):
        cid = client.post(
            CUSTOMERS,
            headers=rep_h,
            json={"nic_number": "900000000401", "full_name": "Ravi Perera"},
        ).get_json()["id"]
        pid = client.post(
            PROPOSALS,
            headers=rep_h,
            json={
                "customer_id": cid,
                "product_id": seed_product_id,
                "proposed_amount": "75000.00",
            },
        ).get_json()["id"]

        client.delete(f"{PRODUCTS}/{seed_product_id}", headers=admin_h)

        # The FK still resolves and the snapshot still reads correctly.
        detail = client.get(f"{PROPOSALS}/{pid}", headers=rep_h)
        assert detail.status_code == 200
        assert detail.get_json()["product_type"] == SEED_PRODUCT_NAME

    def test_a_deleted_name_can_be_reused(self, client, admin_h, seed_product_id):
        client.delete(f"{PRODUCTS}/{seed_product_id}", headers=admin_h)
        # The retired row no longer reserves its name.
        assert _create(client, admin_h, name=SEED_PRODUCT_NAME).status_code == 201


class TestListing:
    def test_detail_reports_proposal_count(self, client, rep_h, admin_h, seed_product_id):
        assert (
            client.get(f"{PRODUCTS}/{seed_product_id}", headers=admin_h).get_json()[
                "proposal_count"
            ]
            == 0
        )
        cid = client.post(
            CUSTOMERS,
            headers=rep_h,
            json={"nic_number": "900000000402", "full_name": "Ravi Perera"},
        ).get_json()["id"]
        client.post(
            PROPOSALS,
            headers=rep_h,
            json={
                "customer_id": cid,
                "product_id": seed_product_id,
                "proposed_amount": "75000.00",
            },
        )
        assert (
            client.get(f"{PRODUCTS}/{seed_product_id}", headers=admin_h).get_json()[
                "proposal_count"
            ]
            == 1
        )

    def test_filter_by_category(self, client, admin_h):
        _create(client, admin_h, name="Agar One", category="agarwood")
        agarwood = client.get(f"{PRODUCTS}?category=agarwood", headers=admin_h).get_json()
        assert [p["name"] for p in agarwood["items"]] == ["Agar One"]
        teak = client.get(f"{PRODUCTS}?category=teak", headers=admin_h).get_json()
        assert [p["name"] for p in teak["items"]] == [SEED_PRODUCT_NAME]

    def test_invalid_category_filter_is_422(self, client, admin_h):
        assert client.get(f"{PRODUCTS}?category=bogus", headers=admin_h).status_code == 422

    def test_filter_by_is_active(self, client, admin_h, seed_product_id):
        _create(client, admin_h, name="Off Sale", is_active=False)
        active = client.get(f"{PRODUCTS}?is_active=true", headers=admin_h).get_json()
        assert [p["id"] for p in active["items"]] == [seed_product_id]
        inactive = client.get(f"{PRODUCTS}?is_active=false", headers=admin_h).get_json()
        assert [p["name"] for p in inactive["items"]] == ["Off Sale"]

    def test_search_matches_name_and_code(self, client, admin_h):
        _create(client, admin_h, name="Coconut Estate Unit")
        by_name = client.get(f"{PRODUCTS}?search=coconut", headers=admin_h).get_json()
        assert [p["name"] for p in by_name["items"]] == ["Coconut Estate Unit"]
        by_code = client.get(
            f"{PRODUCTS}?search={SEED_PRODUCT_CODE}", headers=admin_h
        ).get_json()
        assert [p["product_code"] for p in by_code["items"]] == [SEED_PRODUCT_CODE]

    def test_pagination_envelope(self, client, admin_h):
        for n in range(3):
            _create(client, admin_h, name=f"Bulk Product {n}")
        page = client.get(f"{PRODUCTS}?page=1&per_page=2", headers=admin_h).get_json()
        assert len(page["items"]) == 2
        # 3 created + 1 seeded.
        assert page["pagination"]["total"] == 4
        assert page["pagination"]["pages"] == 2
        assert page["pagination"]["has_next"] is True
        assert page["pagination"]["has_prev"] is False

    def test_per_page_is_capped(self, client, admin_h):
        page = client.get(f"{PRODUCTS}?per_page=9999", headers=admin_h).get_json()
        assert page["pagination"]["per_page"] == 100
