"""Integration tests for the Phase 5 document upload + OCR slice.

The heavy EasyOCR model is never invoked: ``extractor.run_ocr`` is monkeypatched
to return deterministic lines, so the real image preprocessing, field parsing,
file storage/hashing, persistence, and routing all run for real while the model
does not. Uploads are written to a per-test temp directory.
"""

import re
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

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

# Deterministic stand-in for a recognized bank slip.
_OCR_LINES = [
    ("Bank Deposit Slip", 0.99),
    ("Name: Nimal Perera", 0.95),
    ("NIC 912345678V", 0.93),
    ("Account No 1234567890", 0.90),
    ("Amount Rs. 50,000.00", 0.88),
    ("Date 2025-08-16", 0.92),
]


@pytest.fixture(autouse=True)
def uploads_dir(app, tmp_path):
    """Route all stored uploads into a throwaway temp directory."""
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def stub_ocr(monkeypatch):
    """Replace the EasyOCR call with deterministic output."""
    monkeypatch.setattr("app.ai.ocr.extractor.run_ocr", lambda image: list(_OCR_LINES))


@pytest.fixture
def admin_h(token_for, auth_header):
    return auth_header(token_for(ADMIN_EMAIL, ADMIN_PASSWORD))


@pytest.fixture
def rep_h(token_for, auth_header):
    return auth_header(token_for(REP_EMAIL, REP_PASSWORD))


@pytest.fixture
def rep2_h(token_for, auth_header):
    return auth_header(token_for(REP2_EMAIL, REP2_PASSWORD))


def _png_bytes(size=(600, 300)) -> bytes:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline="black")
    draw.text((30, 40), "BANK DEPOSIT SLIP", fill="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_customer(client, headers, nic="900000000001"):
    resp = client.post(
        CUSTOMERS, headers=headers, json={"nic_number": nic, "full_name": "Nimal Perera"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


def _upload(client, headers, customer_id, *, doc_type="bank_slip", content=None, filename="slip.png"):
    data = {
        "customer_id": str(customer_id),
        "doc_type": doc_type,
        "file": (BytesIO(content if content is not None else _png_bytes()), filename),
    }
    return client.post(UPLOAD, headers=headers, data=data, content_type="multipart/form-data")


class TestUpload:
    def test_upload_returns_extracted_fields_and_confidence(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = _upload(client, admin_h, cid)
        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()

        assert body["customer_id"] == cid
        assert body["doc_type"] == "bank_slip"
        assert body["verification_status"] == "pending"
        # mean of the six stubbed confidences.
        assert body["ocr_confidence"] == pytest.approx(0.928, abs=1e-3)

        fields = body["extracted_fields"]["fields"]
        assert fields["nic_number"]["value"] == "912345678V"
        assert fields["name"]["value"] == "Nimal Perera"
        assert fields["amount"]["value"] == "50,000.00"
        assert fields["date"]["value"] == "2025-08-16"
        assert fields["account_no"]["value"] == "1234567890"

    def test_file_stored_with_uuid_name_and_hash(self, client, admin_h, uploads_dir):
        cid = _make_customer(client, admin_h)
        body = _upload(client, admin_h, cid).get_json()

        assert re.fullmatch(r"[0-9a-f]{64}", body["sha256_hash"])
        stored = list(uploads_dir.iterdir())
        assert len(stored) == 1
        assert re.fullmatch(r"[0-9a-f]{32}\.png", stored[0].name)

    def test_hash_matches_stored_file(self, client, admin_h, uploads_dir):
        import hashlib

        cid = _make_customer(client, admin_h)
        body = _upload(client, admin_h, cid).get_json()
        stored = list(uploads_dir.iterdir())[0]
        digest = hashlib.sha256(stored.read_bytes()).hexdigest()
        assert digest == body["sha256_hash"]

    def test_rejects_unsupported_media_type(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = _upload(client, admin_h, cid, content=b"just plain text", filename="note.txt")
        assert resp.status_code == 415
        assert resp.get_json()["error"] == "unsupported_media_type"

    def test_rejects_missing_file(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = client.post(
            UPLOAD,
            headers=admin_h,
            data={"customer_id": str(cid), "doc_type": "bank_slip"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 422

    def test_rejects_invalid_doc_type(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = _upload(client, admin_h, cid, doc_type="passport")
        assert resp.status_code == 422

    def test_upload_for_missing_customer_is_404(self, client, admin_h):
        resp = _upload(client, admin_h, 999999)
        assert resp.status_code == 404

    def test_requires_authentication(self, client):
        resp = _upload(client, {}, 1)
        assert resp.status_code == 401


class TestAccessScope:
    def test_rep_cannot_upload_for_another_reps_customer(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000050")
        resp = _upload(client, rep2_h, cid)
        assert resp.status_code == 403

    def test_rep_can_upload_for_own_customer(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000051")
        assert _upload(client, rep_h, cid).status_code == 201


class TestRetrieve:
    def test_get_document_metadata(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]

        resp = client.get(f"/api/v1/documents/{doc_id}", headers=admin_h)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["id"] == doc_id
        assert "extracted_fields" not in body  # metadata view omits the payload

    def test_get_ocr_result(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        doc_id = _upload(client, admin_h, cid).get_json()["id"]

        resp = client.get(f"/api/v1/documents/{doc_id}/ocr-result", headers=admin_h)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["extracted_fields"]["fields"]["nic_number"]["value"] == "912345678V"

    def test_get_missing_document_is_404(self, client, admin_h):
        assert client.get("/api/v1/documents/999999", headers=admin_h).status_code == 404

    def test_rep_cannot_read_another_reps_document(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000060")
        doc_id = _upload(client, rep_h, cid).get_json()["id"]
        resp = client.get(f"/api/v1/documents/{doc_id}", headers=rep2_h)
        assert resp.status_code == 403
