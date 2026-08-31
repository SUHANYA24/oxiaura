"""Template-based field extraction for fixed-layout documents.

The regex parser in :mod:`app.ai.ocr.extractor` works well on free-form text
(bank slips, printed letters) but struggles on a ruled form: the labels and the
handwritten values sit in separate table cells, so a line-oriented "name:
(.+)" pattern has nothing to latch onto, and OCR run over the whole page mixes
the two columns together.

This module takes the other approach. A *template* records, once, where each
field sits on a reference scan as normalized fractions of the page. At
extraction time a new scan is aligned to that reference with an ORB homography,
each field box is cropped, and OCR runs on the crop alone — so a value is read
from a region that contains nothing but that value.

Output is deliberately shaped like :func:`app.ai.ocr.extractor.parse_fields`
(``raw_text`` / ``mean_confidence`` / ``fields``) so the two strategies are
interchangeable for callers.

Templates are produced by ``ml_training/calibrate_template.py`` and live in
``ml_training/templates/*.json``:

.. code-block:: json

    {
      "name": "proposal_v1",
      "doc_kind": "proposal",
      "reference_image": "app/ai/training/proposal/proposal-1-front.jpeg",
      "reference_size": [1200, 1600],
      "rotate": 0,
      "fields": {"full_name": {"box": [0.22, 0.15, 0.70, 0.05]}}
    }

Box coordinates are ``[x, y, width, height]`` as fractions of the page, which
keeps a template valid across scan resolutions.

``rotate`` is counter-clockwise degrees (0, 90, 180, 270) applied to the scan
*before* alignment, for documents habitually photographed sideways — an ID card
held landscape in a portrait frame. Boxes are then given in the upright frame.
Rotating here rather than re-saving the source file matters: the fraud detectors
read compression artifacts straight from the original bytes, and re-encoding a
scan to straighten it would destroy the very evidence they look for.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import preprocessor

# Alignment needs a minimum number of agreeing keypoints to be trustworthy.
# Below this we fall back to a plain resize and say so, rather than warping the
# page with a homography estimated from noise.
_MIN_MATCHES = 12
_ORB_FEATURES = 5000
# Field crops are upscaled to at least this long edge before OCR — the same
# reasoning as preprocessor._MIN_LONG_EDGE, but a crop is much smaller than a
# page so it needs its own floor.
_CROP_MIN_LONG_EDGE = 480

# Counter-clockwise degrees -> the cv2 constant that achieves it. 0 is a no-op.
_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


class TemplateError(ValueError):
    """Raised when a template file is missing required structure."""


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees counter-clockwise (lossless).

    Used to bring a sideways scan upright before alignment, so field boxes can be
    written in the orientation a human reads.
    """
    if degrees not in _ROTATIONS:
        raise TemplateError(
            f"rotate must be one of {sorted(_ROTATIONS)} (counter-clockwise degrees), "
            f"got {degrees!r}."
        )
    code = _ROTATIONS[degrees]
    return image if code is None else cv2.rotate(image, code)


def load_template(path: str | Path) -> dict:
    """Load and validate a template JSON file.

    :raises TemplateError: if required keys are missing or a box is malformed.
    """
    path = Path(path)
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TemplateError(f"Template not found: {path}")
    except json.JSONDecodeError as exc:
        raise TemplateError(f"Template {path} is not valid JSON: {exc}")

    fields = template.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise TemplateError(f"Template {path} has no 'fields' object.")

    for name, spec in fields.items():
        box = spec.get("box") if isinstance(spec, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise TemplateError(
                f"Field '{name}' in {path} needs a 'box' of [x, y, width, height]."
            )
        if not all(isinstance(value, (int, float)) for value in box):
            raise TemplateError(f"Field '{name}' in {path} has a non-numeric box value.")

    if template.get("rotate", 0) not in _ROTATIONS:
        raise TemplateError(
            f"Template {path} has rotate={template.get('rotate')!r}; expected one of "
            f"{sorted(_ROTATIONS)} (counter-clockwise degrees)."
        )

    template.setdefault("name", path.stem)
    return template


def save_template(template: dict, path: str | Path) -> Path:
    """Write a template to disk with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return path


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, tolerating non-ASCII paths on Windows.

    ``cv2.imread`` silently returns ``None`` for paths it cannot encode, so the
    bytes are read by pathlib and decoded in memory instead.
    """
    path = Path(path)
    buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image at {str(path)!r}.")
    return image


def align_to_template(
    image: np.ndarray, reference: np.ndarray, min_matches: int = _MIN_MATCHES
) -> tuple[np.ndarray, dict]:
    """Warp ``image`` onto ``reference``'s frame using an ORB homography.

    :returns: ``(aligned_image, info)`` where ``info`` carries ``aligned`` (bool),
        the keypoint match count, and the RANSAC inlier count. When alignment is
        not possible the image is simply resized to the reference size and
        ``aligned`` is ``False`` — degraded but still usable, and the caller can
        treat it as lower confidence.
    """
    ref_h, ref_w = reference.shape[:2]
    info = {"aligned": False, "matches": 0, "inliers": 0, "reason": None}

    gray_image = preprocessor.to_grayscale(image)
    gray_reference = preprocessor.to_grayscale(reference)

    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp_image, desc_image = orb.detectAndCompute(gray_image, None)
    kp_reference, desc_reference = orb.detectAndCompute(gray_reference, None)

    if desc_image is None or desc_reference is None:
        info["reason"] = "no ORB descriptors"
        return cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC), info

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(desc_image, desc_reference), key=lambda m: m.distance)
    info["matches"] = len(matches)

    if len(matches) < min_matches:
        info["reason"] = f"only {len(matches)} matches (need {min_matches})"
        return cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC), info

    # Keep the strongest half — weak matches drag the homography off.
    kept = matches[: max(min_matches, len(matches) // 2)]
    source = np.float32([kp_image[m.queryIdx].pt for m in kept]).reshape(-1, 1, 2)
    target = np.float32([kp_reference[m.trainIdx].pt for m in kept]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if homography is None:
        info["reason"] = "homography estimation failed"
        return cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC), info

    info["inliers"] = int(mask.sum()) if mask is not None else 0
    if info["inliers"] < min_matches:
        info["reason"] = f"only {info['inliers']} RANSAC inliers"
        return cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC), info

    info["aligned"] = True
    aligned = cv2.warpPerspective(
        image, homography, (ref_w, ref_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return aligned, info


def crop_field(image: np.ndarray, box, pad: float = 0.004) -> np.ndarray:
    """Crop a normalized ``[x, y, w, h]`` box, padded slightly and clamped."""
    height, width = image.shape[:2]
    x, y, w, h = box

    x0 = int(round((x - pad) * width))
    y0 = int(round((y - pad) * height))
    x1 = int(round((x + w + pad) * width))
    y1 = int(round((y + h + pad) * height))

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8) if image.ndim == 3 else np.zeros((1, 1), np.uint8)
    return image[y0:y1, x0:x1]


def prepare_crop(crop: np.ndarray) -> np.ndarray:
    """Light preprocessing for a single field crop.

    Reuses :mod:`app.ai.ocr.preprocessor`, but deliberately skips two of its
    steps: deskew (the homography already straightened the page, and estimating
    skew from a few words is unreliable) and adaptive threshold (it eats thin
    handwriting strokes — EasyOCR reads grayscale pen better than a binarized
    version of it).
    """
    gray = preprocessor.to_grayscale(crop)
    gray = preprocessor.denoise(gray)
    return preprocessor.resize(gray, min_long_edge=_CROP_MIN_LONG_EDGE)


def extract_fields(
    image: np.ndarray,
    template: dict,
    ocr_fn=None,
    *,
    reference: np.ndarray | None = None,
    align: bool = True,
) -> dict:
    """Read every templated field from ``image``.

    :param image: BGR page image (unaligned is fine — pass ``reference`` to align).
    :param template: as returned by :func:`load_template`.
    :param ocr_fn: callable taking an image and returning ``[(text, confidence)]``;
        defaults to :func:`app.ai.ocr.extractor.run_ocr`. Injectable so tests and
        the evaluation harness can substitute a stub instead of loading EasyOCR.
    :param reference: the template's reference scan, needed for alignment.
    :param align: set ``False`` to treat ``image`` as already aligned.
    :returns: ``{"raw_text", "mean_confidence", "fields", "alignment"}`` — the
        same shape as :func:`app.ai.ocr.extractor.parse_fields`, plus alignment
        diagnostics.
    """
    if ocr_fn is None:
        from . import extractor

        ocr_fn = extractor.run_ocr

    # Bring both frames upright first, so boxes are read in the orientation they
    # were calibrated in and ORB matches features the same way up.
    rotate = int(template.get("rotate", 0))
    if rotate:
        image = rotate_image(image, rotate)
        if reference is not None:
            reference = rotate_image(reference, rotate)

    alignment = {"aligned": False, "matches": 0, "inliers": 0, "reason": "alignment skipped"}
    if align and reference is not None:
        image, alignment = align_to_template(image, reference)

    fields: dict[str, dict] = {}
    all_confidences: list[float] = []
    raw_lines: list[str] = []

    for name, spec in template["fields"].items():
        crop = crop_field(image, spec["box"], pad=float(spec.get("pad", 0.004)))
        lines = ocr_fn(prepare_crop(crop))
        if not lines:
            continue

        text = " ".join(part for part, _ in lines).strip()
        confidences = [float(confidence) for _, confidence in lines]
        if not text:
            continue

        confidence = round(sum(confidences) / len(confidences), 4)
        fields[name] = {"value": text, "confidence": confidence}
        all_confidences.extend(confidences)
        raw_lines.append(f"{name}: {text}")

    mean_confidence = (
        round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0
    )
    return {
        "raw_text": "\n".join(raw_lines),
        "mean_confidence": mean_confidence,
        "fields": fields,
        "alignment": alignment,
    }


def extract_from_path(
    image_path: str | Path, template: dict, ocr_fn=None, template_dir: str | Path | None = None
) -> dict:
    """Convenience wrapper: read the image and its reference, then extract.

    The template's ``reference_image`` is resolved relative to the repository
    root, falling back to ``template_dir`` for a relocated template.
    """
    image = read_image(image_path)

    reference = None
    ref_name = template.get("reference_image")
    if ref_name:
        # app/ai/ocr/template_extractor.py -> repository root is three levels up.
        candidates = [Path(__file__).resolve().parents[3] / ref_name, Path(ref_name)]
        if template_dir:
            candidates.append(Path(template_dir) / ref_name)
        for candidate in candidates:
            if candidate.is_file():
                reference = read_image(candidate)
                break

    return extract_fields(image, template, ocr_fn=ocr_fn, reference=reference)
