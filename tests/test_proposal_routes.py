"""Integration tests for the Phase 9 proposal workflow.

Covers submission, status-filtered listing, access scoping, the
submitted -> rep_review -> ho_review -> approved/rejected state machine with
per-transition role checks, and the notification queued on each stage change.
"""

import pytest

from app.models import Notification
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

CUSTOMERS = "/api/v1/customers"
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


def _make_customer(client, headers, nic="900000000301"):
    resp = client.post(
        CUSTOMERS, headers=headers, json={"nic_number": nic, "full_name": "Sunil Silva"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


def _submit(client, headers, customer_id, **overrides):
    payload = {
        "customer_id": customer_id,
        "proposed_amount": "75000.00",
        "product_type": "Teak Plantation Unit",
    }
    payload.update(overrides)
    return client.post(PROPOSALS, headers=headers, json=payload)


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
        assert body["product_type"] == "Teak Plantation Unit"
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


