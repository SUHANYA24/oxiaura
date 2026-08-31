"""Unit tests for the secure file handler (MIME sniffing, hashing, integrity)."""

from app.utils import file_handler


class TestDetectMime:
    def test_png_magic_bytes(self):
        assert file_handler.detect_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == "image/png"

    def test_jpeg_magic_bytes(self):
        assert file_handler.detect_mime(b"\xff\xd8\xff" + b"\x00" * 12) == "image/jpeg"

    def test_unknown_bytes_return_none(self):
        assert file_handler.detect_mime(b"plain text content here") is None

    def test_too_short_returns_none(self):
        assert file_handler.detect_mime(b"\xff\xd8") is None


class TestIntegrity:
    def test_verify_integrity_detects_tampering(self, tmp_path):
        target = tmp_path / "doc.bin"
        target.write_bytes(b"original")
        digest = file_handler.compute_sha256(b"original")
        assert file_handler.verify_integrity(str(target), digest) is True

        target.write_bytes(b"tampered")
        assert file_handler.verify_integrity(str(target), digest) is False

    def test_verify_integrity_missing_file(self, tmp_path):
        assert file_handler.verify_integrity(str(tmp_path / "nope.bin"), "abc") is False
