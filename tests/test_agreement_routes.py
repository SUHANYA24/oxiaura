"""Integration tests for the Phase 8 agreement slice (generate + verify).

WeasyPrint's native rendering is stubbed via ``html_to_pdf`` so the template,
QR generation, sequencing, storage, access control, and the public verify flow
all run for real without the native PDF engine. PDFs are written to a per-test
temp directory.
"""

from io import BytesIO

import pytest

from app.models import Agreement
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
GENERATE = "/api/v1/agreements/generate"
_STUB_PDF = b"%PDF-1.4 stub agreement"


@pytest.fixture(autouse=True)
def agreements_dir(app, tmp_path):
    app.config["AGREEMENTS_FOLDER"] = str(tmp_path / "agreements")
    return tmp_path / "agreements"


@pytest.fixture(autouse=True)
def stub_pdf(monkeypatch):
    """Skip the native WeasyPrint render; return deterministic PDF bytes."""
    monkeypatch.setattr(
        "app.services.agreement_service.html_to_pdf", lambda html: _STUB_PDF
    )


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


def _make_customer(client, headers, nic="900000000201"):
    resp = client.post(
        CUSTOMERS, headers=headers, json={"nic_number": nic, "full_name": "Nimal Perera"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


def _generate(client, headers, customer_id, **overrides):
    payload = {
        "customer_id": customer_id,
        "investment_amount": "50000.00",
        "duration_months": 12,
        "interest_rate": 8.5,
    }
    payload.update(overrides)
    return client.post(GENERATE, headers=headers, json=payload)


class TestGenerate:
    def test_generate_returns_agreement_with_token_and_pdf(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = _generate(client, admin_h, cid)
        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()

        assert body["customer_id"] == cid
        assert body["agreement_number"].startswith("AGR-")
        assert body["status"] == "pending"
        assert body["investment_amount"] == "50000.00"
        assert body["qr_code_token"]
        assert body["has_pdf"] is True

    def test_pdf_written_to_disk(self, client, admin_h, agreements_dir):
        cid = _make_customer(client, admin_h)
        _generate(client, admin_h, cid)
        stored = list(agreements_dir.iterdir())
        assert len(stored) == 1
        assert stored[0].suffix == ".pdf"
        assert stored[0].read_bytes() == _STUB_PDF

    def test_agreement_numbers_are_sequential(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        first = _generate(client, admin_h, cid).get_json()["agreement_number"]
        second = _generate(client, admin_h, cid).get_json()["agreement_number"]
        assert first != second
        # Same year prefix, incrementing suffix.
        assert first.rsplit("-", 1)[0] == second.rsplit("-", 1)[0]
        assert int(second.rsplit("-", 1)[1]) == int(first.rsplit("-", 1)[1]) + 1

    def test_staff_can_generate(self, client, admin_h, staff_h):
        cid = _make_customer(client, admin_h)
        assert _generate(client, staff_h, cid).status_code == 201

    def test_missing_customer_is_404(self, client, admin_h):
        assert _generate(client, admin_h, 999999).status_code == 404

    def test_invalid_amount_is_422(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        assert _generate(client, admin_h, cid, investment_amount="0").status_code == 422

    def test_missing_field_is_422(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        resp = client.post(
            GENERATE, headers=admin_h, json={"customer_id": cid, "duration_months": 12}
        )
        assert resp.status_code == 422

    def test_requires_authentication(self, client):
        assert client.post(GENERATE, json={}).status_code == 401


class TestAccessScope:
    def test_rep_can_generate_for_own_customer(self, client, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000210")
        assert _generate(client, rep_h, cid).status_code == 201

    def test_rep_cannot_generate_for_another_reps_customer(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000211")
        assert _generate(client, rep2_h, cid).status_code == 403

    def test_rep_only_lists_own_agreements(self, client, admin_h, rep_h):
        own = _make_customer(client, rep_h, nic="900000000212")
        other = _make_customer(client, admin_h, nic="900000000213")
        _generate(client, rep_h, own)
        _generate(client, admin_h, other)

        items = client.get("/api/v1/agreements", headers=rep_h).get_json()["items"]
        assert len(items) == 1
        assert items[0]["customer_id"] == own

    def test_rep_cannot_read_another_reps_agreement(self, client, rep_h, rep2_h):
        cid = _make_customer(client, rep_h, nic="900000000214")
        aid = _generate(client, rep_h, cid).get_json()["id"]
        assert client.get(f"/api/v1/agreements/{aid}", headers=rep2_h).status_code == 403


class TestPdfDownload:
    def test_download_returns_pdf(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        aid = _generate(client, admin_h, cid).get_json()["id"]
        resp = client.get(f"/api/v1/agreements/{aid}/pdf", headers=admin_h)
        assert resp.status_code == 200
        assert resp.mimetype == "application/pdf"
        assert resp.data == _STUB_PDF


class TestPublicVerify:
    def test_valid_token_confirms_authenticity(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        body = _generate(client, admin_h, cid).get_json()
        token = body["qr_code_token"]

        # No auth header — the verify endpoint is public.
        resp = client.get(f"/api/v1/verify/{token}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is True
        assert data["agreement_number"] == body["agreement_number"]
        assert data["customer_name"] == "Nimal Perera"
        assert data["investment_amount"] == "50,000.00"

    def test_tampered_token_fails(self, client, admin_h):
        cid = _make_customer(client, admin_h)
        token = _generate(client, admin_h, cid).get_json()["qr_code_token"]
        encoded, sig = token.rsplit(".", 1)
        forged = f"{encoded}.{'0' * len(sig)}"

        resp = client.get(f"/api/v1/verify/{forged}")
        assert resp.status_code == 404
        assert resp.get_json()["valid"] is False

    def test_unknown_but_signed_token_fails(self, client):
        # A validly-signed token for an agreement that was never stored.
        from app.utils import qr_generator

        with client.application.app_context():
            token = qr_generator.make_token("AGR-2026-999")
        resp = client.get(f"/api/v1/verify/{token}")
        assert resp.status_code == 404
        assert resp.get_json()["valid"] is False

