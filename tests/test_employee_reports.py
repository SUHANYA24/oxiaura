"""Integration tests for the Phase 10 employee KPI + reporting endpoints.

Covers monthly target upsert, the employee KPI list, the auto-increment of
actuals when customers are registered and agreements generated, the dashboard
aggregates, and the filtered CSV export. WeasyPrint is stubbed so agreement
generation (which drives revenue actuals) runs without the native engine.
"""

from datetime import datetime

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

CUSTOMERS = "/api/v1/customers"
EMPLOYEES = "/api/v1/employees"
GENERATE = "/api/v1/agreements/generate"

_NOW = datetime.utcnow()
_MONTH, _YEAR = _NOW.month, _NOW.year


@pytest.fixture(autouse=True)
def agreements_dir(app, tmp_path):
    app.config["AGREEMENTS_FOLDER"] = str(tmp_path / "agreements")


@pytest.fixture(autouse=True)
def stub_pdf(monkeypatch):
    monkeypatch.setattr(
        "app.services.agreement_service.html_to_pdf", lambda html: b"%PDF stub"
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


def _make_customer(client, headers, nic="900000000401"):
    resp = client.post(
        CUSTOMERS, headers=headers, json={"nic_number": nic, "full_name": "Kamala Fonseka"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["id"]


def _generate(client, headers, customer_id, amount="50000.00"):
    resp = client.post(
        GENERATE,
        headers=headers,
        json={
            "customer_id": customer_id,
            "investment_amount": amount,
            "duration_months": 12,
            "interest_rate": 8.5,
        },
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()


def _target_body(**overrides):
    body = {
        "month": _MONTH,
        "year": _YEAR,
        "target_customers": 10,
        "target_revenue": "1000000.00",
    }
    body.update(overrides)
    return body


def _employee_row(client, headers, user_id):
    items = client.get(
        f"{EMPLOYEES}?month={_MONTH}&year={_YEAR}", headers=headers
    ).get_json()["items"]
    return next(r for r in items if r["user_id"] == user_id)


class TestTargets:
    def test_admin_sets_target(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{EMPLOYEES}/{rep_id}/targets", headers=admin_h, json=_target_body()
        )
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["user_id"] == rep_id
        assert body["target_customers"] == 10
        assert body["target_revenue"] == "1000000.00"
        assert body["actual_customers"] == 0

    def test_target_upsert_preserves_actuals(self, client, admin_h, rep_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        _make_customer(client, rep_h, nic="900000000410")  # bumps actual_customers -> 1
        resp = client.post(
            f"{EMPLOYEES}/{rep_id}/targets", headers=admin_h, json=_target_body()
        )
        assert resp.get_json()["actual_customers"] == 1

    def test_missing_user_is_404(self, client, admin_h):
        resp = client.post(
            f"{EMPLOYEES}/999999/targets", headers=admin_h, json=_target_body()
        )
        assert resp.status_code == 404

    def test_invalid_month_is_422(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{EMPLOYEES}/{rep_id}/targets", headers=admin_h, json=_target_body(month=13)
        )
        assert resp.status_code == 422

    def test_unknown_field_is_422(self, client, admin_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{EMPLOYEES}/{rep_id}/targets", headers=admin_h, json=_target_body(x=1)
        )
        assert resp.status_code == 422

    def test_rep_cannot_set_targets(self, client, rep_h, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        resp = client.post(
            f"{EMPLOYEES}/{rep_id}/targets", headers=rep_h, json=_target_body()
        )
        assert resp.status_code == 403

    def test_requires_authentication(self, client, user_id_by_email):
        rep_id = user_id_by_email(REP_EMAIL)
        assert client.post(f"{EMPLOYEES}/{rep_id}/targets", json={}).status_code == 401


class TestEmployeeList:
    def test_lists_all_users_with_kpi_fields(self, client, admin_h):
        items = client.get(EMPLOYEES, headers=admin_h).get_json()["items"]
        assert len(items) == 5  # the five seeded users
        row = items[0]
        for key in ("customer_achievement_pct", "revenue_achievement_pct", "role"):
            assert key in row

    def test_achievement_reflects_target_and_actuals(
        self, client, admin_h, rep_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        client.post(
            f"{EMPLOYEES}/{rep_id}/targets",
            headers=admin_h,
            json=_target_body(target_customers=4),
        )
        _make_customer(client, rep_h, nic="900000000420")
        row = _employee_row(client, admin_h, rep_id)
        assert row["actual_customers"] == 1
        assert row["customer_achievement_pct"] == 25.0

    def test_rep_cannot_list_employees(self, client, rep_h):
        assert client.get(EMPLOYEES, headers=rep_h).status_code == 403


class TestAutoIncrement:
    def test_customer_registration_bumps_actual_customers(
        self, client, admin_h, rep_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        _make_customer(client, rep_h, nic="900000000430")
        _make_customer(client, rep_h, nic="900000000431")
        assert _employee_row(client, admin_h, rep_id)["actual_customers"] == 2

    def test_agreement_generation_bumps_actual_revenue(
        self, client, admin_h, rep_h, user_id_by_email
    ):
        rep_id = user_id_by_email(REP_EMAIL)
        cid = _make_customer(client, rep_h, nic="900000000432")
        _generate(client, rep_h, cid, amount="120000.00")
        assert _employee_row(client, admin_h, rep_id)["actual_revenue"] == "120000.00"


class TestDashboard:
    URL = "/api/v1/reports/dashboard"

    def test_dashboard_shape_and_totals(self, client, admin_h, rep_h):
        cid = _make_customer(client, rep_h, nic="900000000440")
        _generate(client, rep_h, cid, amount="80000.00")

        data = client.get(self.URL, headers=admin_h).get_json()
        assert data["customers"]["total"] == 1
        assert set(data["customers"]["by_status"]) == {"pending", "verified", "flagged"}
        assert data["agreements"]["total"] == 1
        assert data["agreements"]["total_investment"] == "80000.00"
        assert "fraud" in data
        assert isinstance(data["revenue_by_branch"], list)

    def test_staff_can_view_dashboard(self, client, staff_h):
        assert client.get(self.URL, headers=staff_h).status_code == 200

    def test_rep_cannot_view_dashboard(self, client, rep_h):
        assert client.get(self.URL, headers=rep_h).status_code == 403


class TestExport:
    URL = "/api/v1/reports/export"

    def test_export_returns_csv_with_rows(self, client, admin_h, rep_h):
        _make_customer(client, rep_h, nic="900000000450")
        _make_customer(client, rep_h, nic="900000000451")

        resp = client.get(self.URL, headers=admin_h)
        assert resp.status_code == 200
        assert resp.mimetype == "text/csv"
        assert "attachment" in resp.headers["Content-Disposition"]

        lines = resp.get_data(as_text=True).strip().splitlines()
        assert lines[0].startswith("customer_code")
        assert len(lines) == 3  # header + two customers

    def test_export_filtered_by_rep(self, client, admin_h, rep_h, rep2_h, user_id_by_email):
        _make_customer(client, rep_h, nic="900000000460")
        _make_customer(client, rep2_h, nic="900000000461")
        rep_id = user_id_by_email(REP_EMAIL)

        resp = client.get(f"{self.URL}?rep={rep_id}", headers=admin_h)
        lines = resp.get_data(as_text=True).strip().splitlines()
        assert len(lines) == 2  # header + only the one rep's customer

    def test_invalid_date_is_422(self, client, admin_h):
        assert client.get(f"{self.URL}?date_from=not-a-date", headers=admin_h).status_code == 422

    def test_rep_cannot_export(self, client, rep_h):
        assert client.get(self.URL, headers=rep_h).status_code == 403


