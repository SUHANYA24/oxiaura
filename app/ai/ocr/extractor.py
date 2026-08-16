"""Field extraction from document images using EasyOCR (BUILD_SPEC Phase 5).

``run_ocr`` is the single point that touches the heavy EasyOCR model; it is
lazily initialized so importing this module (e.g. during app start-up or tests)
does not load any weights. ``parse_fields`` is pure Python and turns raw OCR
lines into the structured fields the business cares about, each carrying the
confidence of the line it was recognized from.
"""

from __future__ import annotations

import re
import threading

import numpy as np

# --- Lazy EasyOCR reader ---------------------------------------------------
# The Reader downloads/loads model weights on first use, so we build it once,
# on demand, behind a lock rather than at import time.
_reader = None
_reader_lock = threading.Lock()
_LANGUAGES = ["en"]


def get_reader():
    """Return a process-wide EasyOCR ``Reader`` (built on first call)."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr  # imported lazily — heavy dependency

                _reader = easyocr.Reader(_LANGUAGES, gpu=False)
    return _reader


def run_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """Recognize text in ``image`` and return ``(text, confidence)`` per line.

    Confidence is EasyOCR's per-detection score in ``[0, 1]``. This is the seam
    the service depends on; tests substitute a deterministic stub for it so the
    real model never has to run.
    """
    results = get_reader().readtext(image, detail=1, paragraph=False)
    lines: list[tuple[str, float]] = []
    for detection in results:
        # EasyOCR returns [bbox, text, confidence] when detail=1.
        _, text, confidence = detection
        text = (text or "").strip()
        if text:
            lines.append((text, float(confidence)))
    return lines


# --- Field patterns --------------------------------------------------------
# Sri Lankan NIC: old format = 9 digits + V/X, new format = 12 digits.
_NIC_RE = re.compile(r"\b(\d{9}[VXvx]|\d{12})\b")
_DATE_RE = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b")
_CURRENCY_RE = re.compile(r"(?:Rs\.?|LKR|USD|\$)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)
_DECIMAL_AMOUNT_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{2})\b")
_ACCOUNT_RE = re.compile(r"\b(\d{8,16})\b")
_LABELLED_RE = {
    "name": re.compile(r"name\s*[:\-]?\s*(.+)", re.IGNORECASE),
    "address": re.compile(r"address\s*[:\-]?\s*(.+)", re.IGNORECASE),
}


def _first_match(lines, pattern, group=1):
    """Return ``(value, confidence)`` from the first line matching ``pattern``."""
    for text, confidence in lines:
        match = pattern.search(text)
        if match:
            return match.group(group).strip(), round(float(confidence), 4)
    return None


def _extract_account_number(lines, exclude: set[str]):
    """First 8–16 digit run that isn't the NIC or part of a recognized date."""
    for text, confidence in lines:
        if _DATE_RE.search(text):
            continue
        for match in _ACCOUNT_RE.finditer(text):
            value = match.group(1)
            if value not in exclude:
                return value, round(float(confidence), 4)
    return None


def parse_fields(lines: list[tuple[str, float]]) -> dict:
    """Turn OCR ``(text, confidence)`` lines into structured fields.

    :returns: ``{"raw_text", "mean_confidence", "fields"}`` where ``fields`` maps
        a field name to ``{"value", "confidence"}``. Only fields that were found
        are present.
    """
    fields: dict[str, dict] = {}

    nic = _first_match(lines, _NIC_RE)
    if nic:
        fields["nic_number"] = {"value": nic[0].upper(), "confidence": nic[1]}

    for field in ("name", "address"):
        found = _first_match(lines, _LABELLED_RE[field])
        if found and found[0]:
            fields[field] = {"value": found[0], "confidence": found[1]}

    date = _first_match(lines, _DATE_RE)
    if date:
        fields["date"] = {"value": date[0], "confidence": date[1]}

    amount = _first_match(lines, _CURRENCY_RE) or _first_match(lines, _DECIMAL_AMOUNT_RE)
    if amount:
        fields["amount"] = {"value": amount[0], "confidence": amount[1]}

    exclude = {v["value"] for v in fields.values()}
    account = _extract_account_number(lines, exclude)
    if account:
        fields["account_no"] = {"value": account[0], "confidence": account[1]}

    confidences = [c for _, c in lines]
    mean_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "raw_text": "\n".join(text for text, _ in lines),
        "mean_confidence": mean_confidence,
        "fields": fields,
    }
