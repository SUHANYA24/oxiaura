"""Error Level Analysis (ELA) + EXIF metadata fraud signal (BUILD_SPEC Phase 6).

ELA re-compresses the image at a known JPEG quality and measures how much each
region changes: authentic regions compressed once change uniformly, while
spliced/edited regions at a different compression history stand out. The EXIF
check looks for editing-software fingerprints. Combined into a 0–100 suspicion
score (higher = more suspicious). Fully deterministic, no model weights.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops

_JPEG_QUALITY = 90
_ELA_GAIN = 8.0  # amplifies the small recompression differences into 0–100
_SOFTWARE_TAG = 0x0131  # EXIF "Software"
_EDITING_SOFTWARE = ("photoshop", "gimp", "paint.net", "pixlr", "lightroom", "affinity")
_EXIF_EDIT_PENALTY = 40.0


def _ela_component(image: Image.Image) -> float:
    """Return the ELA suspicion component in ``[0, 100]``."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=_JPEG_QUALITY)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = np.asarray(ImageChops.difference(image, recompressed), dtype="float32")
    # 99th-percentile difference emphasises localized edits over uniform noise.
    high = float(np.percentile(diff, 99))
    return min(100.0, (high / 255.0) * 100.0 * _ELA_GAIN)


def _exif_penalty(image: Image.Image) -> tuple[float, str | None]:
    """Penalty for editing-software fingerprints in EXIF metadata."""
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return 0.0, None
    software = str(exif.get(_SOFTWARE_TAG, "")).lower()
    for name in _EDITING_SOFTWARE:
        if name in software:
            return _EXIF_EDIT_PENALTY, f"EXIF Software tag reports '{software}'"
    return 0.0, None


def score_image(file_path: str) -> tuple[float, dict]:
    """Compute the ELA + EXIF suspicion score for the image at ``file_path``.

    :returns: ``(score_0_100, details)``.
    """
    with Image.open(file_path) as opened:
        image = opened.convert("RGB")
        ela = _ela_component(image)
        exif_penalty, exif_note = _exif_penalty(opened)

    score = round(min(100.0, ela + exif_penalty), 2)
    details = {
        "ela_component": round(ela, 2),
        "exif_penalty": exif_penalty,
        "exif_note": exif_note,
    }
    return score, details
