"""Unit tests for the HMAC-signed QR token helpers (BUILD_SPEC Phase 8, §3.8)."""

import pytest

from app.utils import qr_generator


@pytest.fixture(autouse=True)
def _ctx(app):
    """Token helpers read QR_SECRET_KEY from the app config."""
    with app.app_context():
        yield


class TestMakeToken:
    def test_token_has_two_parts(self):
        token = qr_generator.make_token("AGR-2026-001")
        assert token.count(".") == 1

    def test_tokens_are_unique_per_call(self):
        # The random nonce means even the same number yields distinct tokens.
        assert qr_generator.make_token("AGR-2026-001") != qr_generator.make_token(
            "AGR-2026-001"
        )


class TestVerifyToken:
    def test_roundtrip_returns_agreement_number(self):
        token = qr_generator.make_token("AGR-2026-042")
        assert qr_generator.verify_token(token) == "AGR-2026-042"

    def test_tampered_payload_fails(self):
        token = qr_generator.make_token("AGR-2026-001")
        encoded, sig = token.rsplit(".", 1)
        # Flip the signature -> must not verify.
        forged = encoded + "." + ("0" * len(sig))
        assert qr_generator.verify_token(forged) is None

    def test_wrong_secret_fails(self, app):
        token = qr_generator.make_token("AGR-2026-001")
        app.config["QR_SECRET_KEY"] = "a-different-secret"
        assert qr_generator.verify_token(token) is None

    @pytest.mark.parametrize("bad", ["", "no-dot", "not.base64!!", "....."])
    def test_malformed_tokens_return_none(self, bad):
        assert qr_generator.verify_token(bad) is None
