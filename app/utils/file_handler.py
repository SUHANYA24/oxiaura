"""Secure file-upload handling for document ingestion (BUILD_SPEC §3 rule 5).

Responsibilities:
  * Validate the uploaded content-type **server-side** by sniffing magic bytes,
    not by trusting the client-supplied filename/extension or Content-Type.
  * Enforce a maximum size (Flask's ``MAX_CONTENT_LENGTH`` is the first line of
    defence; we re-check defensively here).
  * Store the file under a ``uuid4`` filename in the configured, non-web-served
    upload directory.
  * Compute and return a SHA-256 hash on upload; ``verify_integrity`` re-hashes
    on access to detect tampering.

This module raises service-layer exceptions (``ValidationError`` /
``UnsupportedMediaError``) so routes can translate them to consistent JSON.
It intentionally knows nothing about HTTP request/response objects.
"""

from __future__ import annotations

import hashlib
import os
import uuid

from flask import current_app

from ..services.errors import UnsupportedMediaError, ValidationError

# Detected MIME type -> canonical file extension. This is the allow-list: any
# upload whose sniffed type is not a key here is rejected. OCR operates on
# raster images, so only image formats are accepted in this phase.
_MIME_EXTENSIONS: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

ALLOWED_MIME_TYPES = frozenset(_MIME_EXTENSIONS)

_HASH_CHUNK = 65536


def detect_mime(data: bytes) -> str | None:
    """Return the MIME type inferred from ``data``'s magic bytes, or ``None``.

    Deliberately does not consult the filename — the point is to catch a file
    that lies about its type via extension or client Content-Type header.
    """
    if len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return None


def compute_sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _max_bytes() -> int:
    mb = current_app.config.get("MAX_UPLOAD_MB", 10)
    return int(mb) * 1024 * 1024


def _upload_dir() -> str:
    folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(folder, exist_ok=True)
    return folder


def validate_upload(file_storage) -> tuple[bytes, str]:
    """Read and validate a Werkzeug ``FileStorage``.

    :returns: ``(raw_bytes, detected_mime)``.
    :raises ValidationError: no file provided, empty file, or oversize.
    :raises UnsupportedMediaError: content type not in the allow-list.
    """
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValidationError("No file was provided under the 'file' field.")

    data = file_storage.read()
    if not data:
        raise ValidationError("The uploaded file is empty.")

    max_bytes = _max_bytes()
    if len(data) > max_bytes:
        raise ValidationError(
            f"File exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB."
        )

    mime = detect_mime(data)
    if mime not in ALLOWED_MIME_TYPES:
        allowed = ", ".join(sorted(ALLOWED_MIME_TYPES))
        raise UnsupportedMediaError(
            f"Unsupported file type. Allowed types: {allowed}."
        )
    return data, mime


def store_file(data: bytes, mime: str) -> tuple[str, str]:
    """Persist ``data`` under a random UUID filename.

    :returns: ``(file_path, sha256_hex)`` where ``file_path`` is the stored path.
    """
    extension = _MIME_EXTENSIONS.get(mime, "")
    filename = f"{uuid.uuid4().hex}{extension}"
    path = os.path.join(_upload_dir(), filename)
    with open(path, "wb") as handle:
        handle.write(data)
    return path, compute_sha256(data)


def verify_integrity(file_path: str, expected_sha256: str) -> bool:
    """Re-hash the stored file and compare against the recorded digest.

    Returns ``False`` if the file is missing or its contents have changed.
    """
    if not os.path.isfile(file_path):
        return False
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256
