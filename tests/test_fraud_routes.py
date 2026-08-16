"""Integration tests for the Phase 6 fraud pipeline (report + verify endpoints).

OCR is stubbed (as in the Phase 5 tests). The fraud detectors are stubbed via
their module functions so aggregate scoring, flagging, notification, and the
admin verify flow are exercised deterministically without torch inference.
"""

from io import BytesIO

import pytest
from PIL import Image

from app.models import Notification, User, UserRole
from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    REP_EMAIL,
    REP_PASSWORD,
    REP2_EMAIL,
    REP2_PASSWORD,
)

CUSTOMERS = "/api/v1/customers"
UPLOAD = "/api/v1/documents/upload"


@pytest.fixture(autouse=True)
def uploads_dir(app, tmp_path):
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def stub_ocr(monkeypatch):
    monkeypatch.setattr("app.ai.ocr.extractor.run_ocr", lambda image: [("x", 0.9)])


@pytest.fixture
def set_scores(monkeypatch):
    """Force the three detectors to fixed scores for deterministic aggregates."""

    def _apply(ela=10.0, cnn=0.1, siamese=0.1):
        monkeypatch.setattr(
            "app.ai.fraud.ela_detector.score_image", lambda p: (ela, {"ela_component": ela})
        )
        monkeypatch.setattr(
            "app.ai.fraud.cnn_classifier.predict_tamper_probability", lambda p: cnn
        )
        monkeypatch.setattr(
            "app.ai.fraud.siamese_detector.highest_similarity", lambda p: siamese
        )

    return _apply


@pytest.fixture
def admin_h(token_for, auth_header):
    return auth_header(token_for(ADMIN_EMAIL, ADMIN_PASSWORD))


@pytest.fixture
def rep_h(token_for, auth_header):
    return auth_header(token_for(REP_EMAIL, REP_PASSWORD))


@pytest.fixture
def rep2_h(token_for, auth_header):
    return auth_header(token_for(REP2_EMAIL, REP2_PASSWORD))


def _png():
    buf = BytesIO()
    Image.new("RGB", (120, 120), "white").save(buf, format="PNG")
    return buf.getvalue()


def _make_customer(client, headers, nic="900000000001"):
    resp = client.post(
        CUSTOMERS, headers=headers, json={"nic_number": nic, "full_name": "Nimal Perera"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


def _upload(client, headers, customer_id):
    data = {
        "customer_id": str(customer_id),
        "doc_type": "bank_slip",
        "file": (BytesIO(_png()), "slip.png"),
    }
    resp = client.post(UPLOAD, headers=headers, data=data, content_type="multipart/form-data")
    return resp


class TestFraudReport:
    def test_report_returns_all_subscores_and_verdict(self, client, admin_h, set_scores):
        set_scores(ela=10.0, cnn=0.1, siamese=0.1)  # aggregate = 3 + 4 + 3 = 10
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]

        resp = client.get(f"/api/v1/documents/{doc_id}/fraud-report", headers=admin_h)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ela_score"] == 10.0
        assert body["cnn_fraud_score"] == 0.1
        assert body["siamese_similarity"] == 0.1
        assert body["aggregate_score"] == 10.0
        assert body["is_flagged"] is False
        assert body["verdict"] == "clear"

    def test_high_scores_flag_and_notify_staff(self, client, admin_h, set_scores):
        set_scores(ela=90.0, cnn=0.9, siamese=0.9)  # aggregate = 27 + 36 + 27 = 90
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]

        body = client.get(
            f"/api/v1/documents/{doc_id}/fraud-report", headers=admin_h
        ).get_json()
        assert body["is_flagged"] is True
        assert body["verdict"] == "flagged"
        assert body["flag_reason"]

        # One notification per active admin + head_office_staff user (2 seeded).
        staff_ids = [
            u.id
            for u in User.query.filter(
                User.role.in_([UserRole.admin.value, UserRole.head_office_staff.value]),
                User.is_active.is_(True),
            )
        ]
        notes = Notification.query.filter(Notification.user_id.in_(staff_ids)).all()
        assert len(notes) == len(staff_ids) == 2
        assert "flagged" in notes[0].title.lower()

    def test_clean_upload_creates_no_staff_notification(self, client, admin_h, set_scores):
        set_scores(ela=5.0, cnn=0.05, siamese=0.05)
        cid = _make_customer(client, admin_h)
        _upload(client, admin_h, cid)
        assert Notification.query.count() == 0

    def test_rep_cannot_read_another_reps_report(self, client, rep_h, rep2_h, set_scores):
        set_scores()
        cid = _make_customer(client, rep_h, nic="900000000070")
        doc_id = _upload(client, rep_h, cid).get_json()["id"]
        resp = client.get(f"/api/v1/documents/{doc_id}/fraud-report", headers=rep2_h)
        assert resp.status_code == 403


class TestVerify:
    def test_admin_verifies_document_and_notifies_rep(
        self, client, admin_h, rep_h, set_scores, user_id_by_email
    ):
        set_scores()
        cid = _make_customer(client, rep_h, nic="900000000071")
        doc_id = _upload(client, rep_h, cid).get_json()["id"]

        resp = client.post(
            f"/api/v1/documents/{doc_id}/verify",
            headers=admin_h,
            json={"decision": "verified"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["verification_status"] == "verified"

        rep_id = user_id_by_email(REP_EMAIL)
        assert Notification.query.filter_by(user_id=rep_id).count() == 1

    def test_reject_with_reason(self, client, admin_h, rep_h, set_scores):
        set_scores()
        cid = _make_customer(client, rep_h, nic="900000000072")
        doc_id = _upload(client, rep_h, cid).get_json()["id"]
        resp = client.post(
            f"/api/v1/documents/{doc_id}/verify",
            headers=admin_h,
            json={"decision": "rejected", "reason": "Altered amount field"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["verification_status"] == "rejected"

    def test_verify_is_admin_only(self, client, admin_h, rep_h, set_scores):
        set_scores()
        cid = _make_customer(client, rep_h, nic="900000000073")
        doc_id = _upload(client, rep_h, cid).get_json()["id"]
        resp = client.post(
            f"/api/v1/documents/{doc_id}/verify",
            headers=rep_h,
            json={"decision": "verified"},
        )
        assert resp.status_code == 403

    def test_invalid_decision_is_422(self, client, admin_h, set_scores):
        set_scores()
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]
        resp = client.post(
            f"/api/v1/documents/{doc_id}/verify",
            headers=admin_h,
            json={"decision": "maybe"},
        )
        assert resp.status_code == 422
