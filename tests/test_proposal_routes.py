"""Integration tests for the Phase 9 proposal workflow.

Covers submission, status-filtered listing, access scoping, the
submitted -> rep_review -> ho_review -> approved/rejected state machine with
per-transition role checks, and the notification queued on each stage change.

Also covers the product-catalog linkage: product_id is required, must reference
a live on-sale product, product_type is a server-set snapshot of the product's
name, and the terms stay editable only until head office takes the proposal.
"""

import pytest

from app.models import Notification, Product
from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    REP_EMAIL,
    REP_PASSWORD,
    REP2_EMAIL,
    REP2_PASSWORD,
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


@pytest.fixture
def rep2_h(token_for, auth_header):
    return auth_header(token_for(REP2_EMAIL, REP2_PASSWORD))


def _seed_product_id():
    """Id of the product seeded by conftest, resolved inside the live app context."""
    return Product.query.filter_by(product_code=SEED_PRODUCT_CODE).one().id


def _make_customer(client, headers, nic="900000000301"):
    resp = client.post(
        CUSTOMERS, headers=headers, json={"nic_number": nic, "full_name": "Sunil Silva"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


def _submit(client, headers, customer_id, **overrides):
    payload = {
        "customer_id": customer_id,
        "product_id": _seed_product_id(),
        "proposed_amount": "75000.00",
    }
    payload.update(overrides)
    return client.post(PROPOSALS, headers=headers, json=payload)


def _revise(client, headers, proposal_id, **body):
    return client.put(f"{PROPOSALS}/{proposal_id}", headers=headers, json=body)


def _advance(client, headers, proposal_id, **body):
    return client.put(
        f"{PROPOSALS}/{proposal_id}/advance", headers=headers, json=body
    )


class TestSubmit:
    def test_submit_returns_proposal_in_submitted_stage(self, client, rep_h):
        cid = _make_customer(client, rep_h)
        resp = _submit(client, rep_h, cid)
        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()
        assert body["customer_id"] == cid
        assert body["workflow_status"] == "submitted"
        assert body["proposed_amount"] == "75000.00"
        # product_type is the server-set snapshot of the selected product's name.
        assert body["product_id"] == _seed_product_id()
        assert body["product_type"] == SEED_PRODUCT_NAME
        assert body["sales_rep_id"]

    def test_missing_customer_is_404(self, client, admin_h):
        assert _submit(client, admin_h, 999999).status_code == 404

    def test_invalid_amount_is_422(self, client, rep_h):
        cid = _make_customer(client, rep_h)
        assert _submit(client, rep_h, cid, proposed_amount="0").status_code == 422

    def test_missing_field_is_422(self, client, rep_h):
        cid = _make_customer(client, rep_h)
        resp = client.post(PROPOSALS, headers=rep_h, json={"customer_id": cid})
        assert resp.status_code == 422

    def test_unknown_field_is_422(self, client, rep_h):
        cid = _make_customer(client, rep_h)
        assert _submit(client, rep_h, cid, hacker="x").status_code == 422

    def test_requires_authentication(self, client):
        assert client.post(PROPOSALS, json={}).status_code == 401

    def test_rep_cannot_submit_for_another_reps_customer(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000302")
        assert _submit(client, rep2_h, cid).status_code == 403


class TestListScope:
    def test_rep_only_lists_own_proposals(self, client, admin_h, rep_h):
        own = _make_customer(client, rep_h, nic="900000000303")
        other = _make_customer(client, admin_h, nic="900000000304")
        _submit(client, rep_h, own)
        _submit(client, admin_h, other)

        items = client.get(PROPOSALS, headers=rep_h).get_json()["items"]
        assert len(items) == 1
        assert items[0]["customer_id"] == own

    def test_filter_by_status(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000305")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)  # -> rep_review

        submitted = client.get(f"{PROPOSALS}?status=submitted", headers=rep_h)
        assert submitted.get_json()["items"] == []
        review = client.get(f"{PROPOSALS}?status=rep_review", headers=rep_h)
        assert len(review.get_json()["items"]) == 1

    def test_invalid_status_filter_is_422(self, client, rep_h):
        assert client.get(f"{PROPOSALS}?status=bogus", headers=rep_h).status_code == 422

    def test_rep_cannot_read_another_reps_proposal(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000306")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert client.get(f"{PROPOSALS}/{pid}", headers=rep2_h).status_code == 403


class TestWorkflow:
    def test_full_happy_path_to_approved(self, client, rep_h, staff_h):
        cid = _make_customer(client, rep_h, nic="900000000310")
        pid = _submit(client, rep_h, cid).get_json()["id"]

        # Owning rep moves it through the two review stages.
        r1 = _advance(client, rep_h, pid)
        assert r1.status_code == 200
        assert r1.get_json()["workflow_status"] == "rep_review"

        r2 = _advance(client, rep_h, pid)
        assert r2.get_json()["workflow_status"] == "ho_review"

        # Head office makes the final decision.
        r3 = _advance(client, staff_h, pid, decision="approved")
        assert r3.status_code == 200
        assert r3.get_json()["workflow_status"] == "approved"

    def test_ho_review_can_be_rejected(self, client, rep_h, staff_h):
        cid = _make_customer(client, rep_h, nic="900000000311")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)
        _advance(client, rep_h, pid)
        r = _advance(client, staff_h, pid, decision="rejected")
        assert r.get_json()["workflow_status"] == "rejected"

    def test_admin_may_advance_any_stage(self, client, admin_h):
        cid = _make_customer(client, admin_h, nic="900000000312")
        pid = _submit(client, admin_h, cid).get_json()["id"]
        assert _advance(client, admin_h, pid).get_json()["workflow_status"] == "rep_review"
        assert _advance(client, admin_h, pid).get_json()["workflow_status"] == "ho_review"
        assert (
            _advance(client, admin_h, pid, decision="approved").get_json()[
                "workflow_status"
            ]
            == "approved"
        )

    def test_staff_cannot_start_rep_review(self, client, rep_h, staff_h):
        # submitted -> rep_review is a sales_rep (or admin) transition.
        cid = _make_customer(client, rep_h, nic="900000000313")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert _advance(client, staff_h, pid).status_code == 403

    def test_rep_cannot_finalize_ho_review(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000314")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)  # rep_review
        _advance(client, rep_h, pid)  # ho_review
        assert _advance(client, rep_h, pid, decision="approved").status_code == 403

    def test_decision_required_at_ho_review(self, client, rep_h, staff_h):
        cid = _make_customer(client, rep_h, nic="900000000315")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)
        _advance(client, rep_h, pid)
        assert _advance(client, staff_h, pid).status_code == 422

    def test_cannot_advance_terminal_proposal(self, client, rep_h, staff_h):
        cid = _make_customer(client, rep_h, nic="900000000316")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)
        _advance(client, rep_h, pid)
        _advance(client, staff_h, pid, decision="approved")
        assert _advance(client, staff_h, pid, decision="rejected").status_code == 422

    def test_rep_cannot_advance_another_reps_proposal(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000317")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert _advance(client, rep2_h, pid).status_code == 403


class TestNotifications:
    def test_reaching_ho_review_notifies_rep(self, client, app, rep_h, user_id_by_email):
        cid = _make_customer(client, rep_h, nic="900000000320")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)  # rep_review (notifies rep)
        _advance(client, rep_h, pid)  # ho_review (notifies rep + staff)

        rep_id = user_id_by_email(REP_EMAIL)
        with app.app_context():
            rep_notes = Notification.query.filter_by(user_id=rep_id).all()
        # One per stage change for the owning rep.
        assert len(rep_notes) == 2
        assert any("ho_review" in n.title for n in rep_notes)


def _create_product(client, admin_h, name, **overrides):
    """Create a second catalog product as admin, returning its id."""
    payload = {
        "name": name,
        "category": "agarwood",
        "min_investment": "100000.00",
        "duration_months": 240,
        "interest_rate": 14.0,
    }
    payload.update(overrides)
    resp = client.post(PRODUCTS, headers=admin_h, json=payload)
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


class TestProductSelection:
    def test_product_id_is_required(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000330")
        resp = client.post(
            PROPOSALS,
            headers=rep_h,
            json={"customer_id": cid, "proposed_amount": "75000.00"},
        )
        assert resp.status_code == 422
        assert "product_id" in resp.get_json()["messages"]

    def test_unknown_product_is_422(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000331")
        assert _submit(client, rep_h, cid, product_id=999999).status_code == 422

    def test_client_cannot_supply_product_type(self, client, rep_h):
        # product_type is server-owned now, so it is an unknown input field.
        cid = _make_customer(client, rep_h, nic="900000000332")
        assert _submit(client, rep_h, cid, product_type="Made Up").status_code == 422

    def test_inactive_product_cannot_be_selected(self, client, admin_h, rep_h):
        pid_product = _seed_product_id()
        client.put(f"{PRODUCTS}/{pid_product}", headers=admin_h, json={"is_active": False})
        cid = _make_customer(client, rep_h, nic="900000000333")
        resp = _submit(client, rep_h, cid)
        assert resp.status_code == 422
        assert "not currently available" in resp.get_json()["message"]

    def test_deleted_product_cannot_be_selected(self, client, admin_h, rep_h):
        client.delete(f"{PRODUCTS}/{_seed_product_id()}", headers=admin_h)
        cid = _make_customer(client, rep_h, nic="900000000334")
        assert _submit(client, rep_h, cid).status_code == 422

    def test_snapshot_survives_a_product_rename(self, client, admin_h, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000335")
        pid = _submit(client, rep_h, cid).get_json()["id"]

        renamed = client.put(
            f"{PRODUCTS}/{_seed_product_id()}",
            headers=admin_h,
            json={"name": "Teak Unit (2026 terms)"},
        )
        assert renamed.status_code == 200

        detail = client.get(f"{PROPOSALS}/{pid}", headers=rep_h).get_json()
        # The snapshot records what was proposed; the nested product reflects now.
        assert detail["product_type"] == SEED_PRODUCT_NAME
        assert detail["product"]["name"] == "Teak Unit (2026 terms)"

    def test_detail_includes_product_summary(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000336")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        product = client.get(f"{PROPOSALS}/{pid}", headers=rep_h).get_json()["product"]
        assert product["product_code"] == SEED_PRODUCT_CODE
        assert product["category"] == "teak"
        assert product["is_active"] is True


class TestRevise:
    def test_rep_can_change_product_while_submitted(self, client, admin_h, rep_h):
        other = _create_product(client, admin_h, "Agarwood Growth Unit")
        cid = _make_customer(client, rep_h, nic="900000000340")
        pid = _submit(client, rep_h, cid).get_json()["id"]

        resp = _revise(client, rep_h, pid, product_id=other)
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["product_id"] == other
        # The snapshot is re-taken for the newly chosen product.
        assert body["product_type"] == "Agarwood Growth Unit"

    def test_amount_and_notes_are_editable(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000341")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        body = _revise(
            client, rep_h, pid, proposed_amount="90000.00", notes="revised"
        ).get_json()
        assert body["proposed_amount"] == "90000.00"
        assert body["notes"] == "revised"

    def test_editable_during_rep_review(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000342")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)  # -> rep_review
        assert _revise(client, rep_h, pid, proposed_amount="80000.00").status_code == 200

    def test_locked_at_ho_review(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000343")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)  # rep_review
        _advance(client, rep_h, pid)  # ho_review
        resp = _revise(client, rep_h, pid, proposed_amount="80000.00")
        assert resp.status_code == 422
        assert "no longer be edited" in resp.get_json()["message"]

    def test_locked_once_terminal(self, client, rep_h, staff_h):
        cid = _make_customer(client, rep_h, nic="900000000344")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        _advance(client, rep_h, pid)
        _advance(client, rep_h, pid)
        _advance(client, staff_h, pid, decision="approved")
        assert _revise(client, rep_h, pid, notes="too late").status_code == 422

    def test_other_rep_cannot_revise(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000345")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert _revise(client, rep2_h, pid, notes="not mine").status_code == 403

    def test_admin_may_revise(self, client, admin_h, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000346")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert _revise(client, admin_h, pid, notes="ho note").status_code == 200

    def test_cannot_revise_onto_an_inactive_product(self, client, admin_h, rep_h):
        other = _create_product(
            client, admin_h, "Retired Unit", is_active=False
        )
        cid = _make_customer(client, rep_h, nic="900000000347")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert _revise(client, rep_h, pid, product_id=other).status_code == 422

    def test_unknown_field_is_422(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000348")
        pid = _submit(client, rep_h, cid).get_json()["id"]
        assert _revise(client, rep_h, pid, customer_id=1).status_code == 422

    def test_missing_proposal_is_404(self, client, admin_h):
        assert _revise(client, admin_h, 999999, notes="x").status_code == 404

    def test_requires_authentication(self, client):
        assert client.put(f"{PROPOSALS}/1", json={}).status_code == 401


