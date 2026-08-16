"""Agreement service (BUILD_SPEC Phase 8).

Flow: validate customer access -> create the agreement row (sequenced
``agreement_number``) -> mint an HMAC-signed QR token -> render the Jinja
letterhead to a PDF (WeasyPrint) with the QR embedded -> store the PDF and
persist its path. A public verify step re-checks the token signature and
returns the agreement's authenticity details.

The WeasyPrint call is isolated in :func:`html_to_pdf` so tests can stub the
native rendering (as OCR/fraud are stubbed elsewhere) while the template,
QR generation, storage, sequencing, and access control all run for real.
Business logic only — no Flask HTTP types leak out (services raise
``ServiceError``; routes translate them).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import uuid
from datetime import datetime

from flask import current_app, render_template, url_for
from sqlalchemy import select

from ..extensions import db
from ..models import Agreement, AgreementStatus, Customer, User
from ..utils import qr_generator
from . import customer_service, employee_service
from .errors import NotFoundError

logger = logging.getLogger(__name__)

_NUMBER_PREFIX = "AGR"
_MAX_PER_PAGE = 100
_DEFAULT_PER_PAGE = 20


def _next_agreement_number() -> str:
    """Return the next sequential number for the current year (``AGR-2026-001``)."""
    year = datetime.utcnow().year
    prefix = f"{_NUMBER_PREFIX}-{year}-"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    rows = (
        db.session.query(Agreement.agreement_number)
        .filter(Agreement.agreement_number.like(f"{prefix}%"))
        .all()
    )
    highest = 0
    for (number,) in rows:
        match = pattern.match(number or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def build_qr_data_uri(token: str) -> str:
    """Render ``token`` (as a verification URL when possible) to a PNG data URI."""
    import qrcode  # lazy: keeps import cost off the request path until needed

    try:
        data = url_for("agreements.verify_agreement", token=token, _external=True)
    except Exception:  # noqa: BLE001 - outside a request context; encode the token
        data = token

    img = qrcode.make(data)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def html_to_pdf(html: str) -> bytes:
    """Convert rendered HTML to PDF bytes (WeasyPrint).

    Isolated as the single seam over the native PDF engine so it can be stubbed
    in tests. ``base_url`` lets the engine resolve any relative assets.
    """
    from weasyprint import HTML  # lazy: native libs only needed at render time

    return HTML(string=html, base_url=current_app.root_path).write_pdf()


def _store_pdf(pdf_bytes: bytes) -> str:
    """Write PDF bytes under ``AGREEMENTS_FOLDER`` with a UUID name; return the path."""
    folder = current_app.config["AGREEMENTS_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{uuid.uuid4().hex}.pdf")
    with open(path, "wb") as handle:
        handle.write(pdf_bytes)
    return path


def _render_pdf(agreement: Agreement, customer: Customer, creator: User) -> bytes:
    """Render the agreement template (with embedded QR) to PDF bytes."""
    qr_data_uri = build_qr_data_uri(agreement.qr_code_token)
    html = render_template(
        "agreement_template.html",
        agreement=agreement,
        customer=customer,
        creator=creator,
        investment_amount=f"{agreement.investment_amount:,.2f}",
        issued_on=datetime.utcnow().strftime("%Y-%m-%d"),
        qr_data_uri=qr_data_uri,
    )
    return html_to_pdf(html)


def generate_agreement(data: dict, current_user: User) -> Agreement:
    """Create an agreement, generate its QR token, and render + store the PDF.

    :raises NotFoundError / ForbiddenError: from the customer scope check
        (a sales_rep may only generate for their own customers).
    """
    # Enforce existence + row-level access on the customer.
    customer: Customer = customer_service.get_customer(data["customer_id"], current_user)

    agreement = Agreement(
        customer_id=customer.id,
        created_by=current_user.id,
        agreement_number=_next_agreement_number(),
        investment_amount=data["investment_amount"],
        duration_months=data["duration_months"],
        interest_rate=data["interest_rate"],
        status=AgreementStatus.pending,
    )
    agreement.qr_code_token = qr_generator.make_token(agreement.agreement_number)
    db.session.add(agreement)
    db.session.flush()  # assign id before rendering / logging

    pdf_bytes = _render_pdf(agreement, customer, current_user)
    agreement.pdf_path = _store_pdf(pdf_bytes)

    # KPI: credit the customer's assigned rep with this agreement's revenue.
    employee_service.record_revenue(customer.assigned_rep_id, agreement.investment_amount)

    db.session.commit()
    logger.info(
        "Agreement %s generated for customer %s by user %s",
        agreement.agreement_number,
        customer.customer_code,
        current_user.id,
    )
    return agreement


def _scoped_query(current_user: User):
    """Base select for agreements visible to ``current_user``."""
    stmt = select(Agreement)
    # Full-access roles see everything; a sales_rep only sees agreements for
    # customers assigned to them (row-level scoping, per BUILD_SPEC §3).
    if not customer_service._has_full_access(current_user):
        stmt = stmt.join(Customer, Agreement.customer_id == Customer.id).where(
            Customer.assigned_rep_id == current_user.id
        )
    return stmt


def list_agreements(
    current_user: User, *, page: int = 1, per_page: int = _DEFAULT_PER_PAGE
):
    """Return a paginated list of agreements scoped to ``current_user``."""
    page = max(page, 1)
    per_page = min(max(per_page, 1), _MAX_PER_PAGE)
    stmt = _scoped_query(current_user).order_by(Agreement.id.desc())
    return db.paginate(stmt, page=page, per_page=per_page, error_out=False)


def get_agreement(agreement_id: int, current_user: User) -> Agreement:
    """Fetch one agreement, enforcing access via its customer's scope."""
    agreement = db.session.get(Agreement, agreement_id)
    if agreement is None:
        raise NotFoundError("Agreement not found.")
    # Reuse the customer access rules; raises Forbidden/NotFound as appropriate.
    customer_service.get_customer(agreement.customer_id, current_user)
    return agreement


def get_pdf_path(agreement_id: int, current_user: User) -> str:
    """Return the on-disk PDF path for an agreement, if it exists."""
    agreement = get_agreement(agreement_id, current_user)
    if not agreement.pdf_path or not os.path.exists(agreement.pdf_path):
        raise NotFoundError("Agreement PDF is not available.")
    return agreement.pdf_path


def verify_agreement(token: str) -> dict:
    """Public authenticity check for a QR token.

    A forged or tampered token fails the HMAC check and returns ``valid: False``
    without touching the database. A valid signature is then matched against a
    stored agreement; only minimal, non-sensitive fields are returned.
    """
    agreement_number = qr_generator.verify_token(token)
    if agreement_number is None:
        return {"valid": False, "reason": "Invalid or tampered token."}

    agreement = Agreement.query.filter_by(qr_code_token=token).first()
    if agreement is None:
        return {"valid": False, "reason": "No matching agreement."}

    return {
        "valid": True,
        "agreement_number": agreement.agreement_number,
        "customer_name": agreement.customer.full_name if agreement.customer else None,
        "investment_amount": f"{agreement.investment_amount:,.2f}",
        "duration_months": agreement.duration_months,
        "status": agreement.status.value,
        "signed_at": agreement.signed_at.isoformat() if agreement.signed_at else None,
    }

