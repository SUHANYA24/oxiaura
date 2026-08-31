"""Unit tests for password hashing and the token blocklist helpers."""

import bcrypt

from app.utils.security import BCRYPT_ROUNDS, hash_password, verify_password
from app.utils import token_blocklist


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("Secret@123")
        assert hashed != "Secret@123"
        assert hashed.startswith("$2")  # bcrypt identifier

    def test_hash_uses_cost_factor_12(self):
        hashed = hash_password("Secret@123")
        # bcrypt hash format: $2b$<cost>$<salt+digest>
        cost = int(hashed.split("$")[2])
        assert cost == BCRYPT_ROUNDS == 12

    def test_verify_accepts_correct_password(self):
        hashed = hash_password("Secret@123")
        assert verify_password("Secret@123", hashed) is True

    def test_verify_rejects_wrong_password(self):
        hashed = hash_password("Secret@123")
        assert verify_password("wrong", hashed) is False

    def test_salts_make_hashes_unique(self):
        assert hash_password("same") != hash_password("same")

    def test_verify_handles_malformed_hash_gracefully(self):
        assert verify_password("anything", "not-a-real-hash") is False

    def test_verify_handles_empty_hash(self):
        assert verify_password("anything", "") is False

    def test_password_over_72_bytes_is_truncated_not_erroring(self):
        # bcrypt 4.x raises on >72 bytes; the helper must truncate defensively.
        long_pw = "a" * 100
        hashed = hash_password(long_pw)
        assert verify_password(long_pw, hashed) is True
        # First 72 bytes matching is enough for bcrypt to accept.
        assert verify_password("a" * 72, hashed) is True


class TestBlocklist:
    def test_add_and_check(self):
        token_blocklist.clear()
        assert token_blocklist.is_blocklisted("jti-1") is False
        token_blocklist.add_to_blocklist("jti-1", expires_at=None)
        assert token_blocklist.is_blocklisted("jti-1") is True

    def test_expired_entries_are_pruned(self):
        token_blocklist.clear()
        token_blocklist.add_to_blocklist("old-jti", expires_at=1.0)  # far in past
        assert token_blocklist.is_blocklisted("old-jti") is False

    def test_clear_empties_the_store(self):
        token_blocklist.add_to_blocklist("jti-x")
        token_blocklist.clear()
        assert token_blocklist.is_blocklisted("jti-x") is False
