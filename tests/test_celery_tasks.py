"""Tests for the Phase 7 async processing plumbing.

Celery runs eagerly under ``TestConfig`` (``task_always_eager``), so enqueuing
executes the task inline against the in-memory DB — no Redis broker or worker
needed. These tests assert the upload endpoint returns a job id immediately and
that the queued job produces OCR + fraud results that appear when polled.
"""

from io import BytesIO

import pytest
from PIL import Image

from app.extensions import db
from app.models import Document, FraudLog
from app.tasks.celery_tasks import process_document
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD

CUSTOMERS = "/api/v1/customers"
UPLOAD = "/api/v1/documents/upload"

_OCR_LINES = [
    ("Name: Nimal Perera", 0.95),
    ("NIC 912345678V", 0.93),
    ("Amount Rs. 50,000.00", 0.88),
]


@pytest.fixture(autouse=True)
def uploads_dir(app, tmp_path):
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def stub_ocr(monkeypatch):
    monkeypatch.setattr("app.ai.ocr.extractor.run_ocr", lambda image: list(_OCR_LINES))


@pytest.fixture
def admin_h(token_for, auth_header):
    return auth_header(token_for(ADMIN_EMAIL, ADMIN_PASSWORD))


def _png():
    buf = BytesIO()
    Image.new("RGB", (120, 120), "white").save(buf, format="PNG")
    return buf.getvalue()


def _make_customer(client, headers, nic="900000000090"):
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
    return client.post(UPLOAD, headers=headers, data=data, content_type="multipart/form-data")


class TestUploadEnqueues:
    def test_upload_accepts_and_returns_job_id(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = _upload(client, admin_h, cid)

        assert resp.status_code == 202
        body = resp.get_json()
        assert body["status"] == "processing"
        assert body["task_id"]
        assert body["id"]

    def test_document_row_exists_immediately(self, client, admin_h, app):
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]
        with app.app_context():
            assert db.session.get(Document, doc_id) is not None

    def test_processing_populates_ocr_and_fraud(self, client, admin_h, app):
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]

        # Eager execution means the job has already run by the time we poll.
        with app.app_context():
            doc = db.session.get(Document, doc_id)
            assert doc.ocr_confidence is not None
            assert doc.extracted_fields["fields"]["nic_number"]["value"] == "912345678V"
            assert FraudLog.query.filter_by(document_id=doc_id).count() == 1


class TestProcessDocumentTask:
    def test_task_returns_summary(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]

        # Re-running the task on the stored document is idempotent and lets us
        # inspect the JSON-serializable summary it returns for job polling.
        result = process_document.apply(args=(doc_id,)).get()
        assert result["document_id"] == doc_id
        assert result["status"] == "processed"
        assert result["ocr_confidence"] is not None

    def test_task_handles_missing_document(self, app):
        with app.app_context():
            result = process_document.apply(args=(999999,)).get()
        assert result["status"] == "not_found"


class TestJobStatusEndpoint:
    def test_job_status_returns_state(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        task_id = _upload(client, admin_h, cid).get_json()["task_id"]

        resp = client.get(f"/api/v1/documents/jobs/{task_id}", headers=admin_h)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["task_id"] == task_id
        assert "state" in body

    def test_job_status_requires_auth(self, client):
        resp = client.get("/api/v1/documents/jobs/whatever")
        assert resp.status_code == 401
