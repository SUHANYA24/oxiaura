"""Build the fraud training set from authentic scans (BUILD_SPEC Phase 6b).

You cannot train a tamper classifier on four photographs, and collecting real
forgeries is slow. This script takes the authentic scans you drop into
``app/ai/training/<kind>/`` and synthesizes both classes from them:

* **genuine** — the scan with mild capture-style variation (small rotation and
  perspective, brightness/contrast/gamma jitter, noise, one JPEG save).
* **tampered** — the same variation applied *on top of* a forgery operation:

  ``copymove``   a text region duplicated elsewhere on the page
  ``splice``     a region pasted in from a different document
  ``textpatch``  a field value painted out and rewritten with new digits
  ``requant``    a region resaved at a much lower JPEG quality

The capture-style variation is applied identically to both classes on purpose.
If it were applied only to one, the classifier would learn the augmentation
instead of the forgery and score ~1.0 on validation while being useless on real
uploads.

The train/validation split is grouped **by source image**: every variant of a
given scan lands in the same split. With a dataset this small, splitting by
variant instead would put near-duplicates on both sides and report accuracy
that does not exist.

Usage::

    python -m ml_training.prepare_dataset                 # build/refresh
    python -m ml_training.prepare_dataset --clean         # wipe first
    python -m ml_training.prepare_dataset --aug-per-image 10 --tamper-per-image 10
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import string
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import config

# JPEG quality range for the final save — a plausible spread for phone photos
# and scanners. Applied to both classes.
_SAVE_QUALITY = (86, 96)
# Quality used for the localized recompression in the ``requant`` forgery; far
# enough below _SAVE_QUALITY to leave a detectable discontinuity.
_REQUANT_QUALITY = (35, 62)


# --- Region finding ---------------------------------------------------------
def find_text_regions(bgr: np.ndarray, limit: int = 40) -> list[tuple[int, int, int, int]]:
    """Return candidate ``(x, y, w, h)`` boxes that contain ink.

    Forgeries are only interesting where there is content: a patch pasted onto
    blank paper teaches the classifier nothing about altered fields. A
    morphological gradient highlights strokes, closing joins them into lines,
    and the resulting contours approximate text blocks.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Keep line-ish blocks: wide enough to be text, not a full-page blob.
        if w >= 0.04 * width and 0.01 * height <= h <= 0.25 * height:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:limit]


def _patch_rect(
    rng: random.Random, shape, regions: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    """Choose a patch rectangle, biased towards a detected text region.

    The size is clamped to a fraction of the page so the forgery is local —
    large enough to survive the 224x224 resize, small enough to be a plausible
    field-level edit rather than a replaced page.
    """
    height, width = shape[:2]
    patch_w = int(rng.uniform(0.10, 0.26) * width)
    patch_h = int(rng.uniform(0.035, 0.14) * height)
    patch_w = max(16, min(patch_w, width - 2))
    patch_h = max(12, min(patch_h, height - 2))

    if regions and rng.random() < 0.85:
        rx, ry, rw, rh = rng.choice(regions)
        cx, cy = rx + rw // 2, ry + rh // 2
        x = cx - patch_w // 2 + int(rng.uniform(-0.02, 0.02) * width)
        y = cy - patch_h // 2 + int(rng.uniform(-0.02, 0.02) * height)
    else:
        x = rng.randrange(0, max(1, width - patch_w))
        y = rng.randrange(0, max(1, height - patch_h))

    x = max(0, min(x, width - patch_w))
    y = max(0, min(y, height - patch_h))
    return x, y, patch_w, patch_h


def _blend(canvas: np.ndarray, patch: np.ndarray, x: int, y: int, feather: int = 5) -> None:
    """Alpha-blend ``patch`` into ``canvas`` at ``(x, y)`` with soft edges.

    A hard paste leaves a one-pixel step that any classifier finds instantly;
    feathering makes the synthetic forgery behave more like a real one.
    """
    h, w = patch.shape[:2]
    if feather > 0 and h > 2 * feather and w > 2 * feather:
        core = np.ones((h - 2 * feather, w - 2 * feather), np.float32)
        mask = cv2.copyMakeBorder(
            core, feather, feather, feather, feather, cv2.BORDER_CONSTANT, value=0
        )
        mask = cv2.GaussianBlur(mask, (2 * feather + 1, 2 * feather + 1), 0)
    else:
        mask = np.ones((h, w), np.float32)

    alpha = mask[..., None]
    target = canvas[y : y + h, x : x + w].astype(np.float32)
    blended = alpha * patch.astype(np.float32) + (1.0 - alpha) * target
    canvas[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)


# --- Forgery operations -----------------------------------------------------
def op_copymove(bgr: np.ndarray, rng: random.Random, regions) -> np.ndarray:
    """Duplicate one text region somewhere else on the same page."""
    out = bgr.copy()
    sx, sy, w, h = _patch_rect(rng, out.shape, regions)
    patch = out[sy : sy + h, sx : sx + w].copy()

    height, width = out.shape[:2]
    for _ in range(12):  # try to land somewhere that does not overlap the source
        dx = rng.randrange(0, max(1, width - w))
        dy = rng.randrange(0, max(1, height - h))
        if abs(dx - sx) > w // 2 or abs(dy - sy) > h // 2:
            break
    _blend(out, patch, dx, dy)
    return out


def op_splice(bgr: np.ndarray, rng: random.Random, regions, donor: np.ndarray) -> np.ndarray:
    """Paste a region taken from a *different* document into this one."""
    out = bgr.copy()
    dx, dy, w, h = _patch_rect(rng, out.shape, regions)

    donor_regions = find_text_regions(donor)
    sx, sy, sw, sh = _patch_rect(rng, donor.shape, donor_regions)
    patch = donor[sy : sy + sh, sx : sx + sw]
    if patch.size == 0:
        return out
    patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_CUBIC)
    _blend(out, patch, dx, dy)
    return out


def op_textpatch(bgr: np.ndarray, rng: random.Random, regions) -> np.ndarray:
    """Paint out a field value and write a different one in its place.

    This is the forgery the business actually cares about — an altered NIC
    number or deposit amount — so it is worth simulating properly: the original
    ink is inpainted away, and the replacement is drawn in ink sampled from the
    region it replaces.
    """
    out = bgr.copy()
    x, y, w, h = _patch_rect(rng, out.shape, regions)
    roi = out[y : y + h, x : x + w]
    if roi.size == 0:
        return out

    # Ink colour = the darkest decile of the region being overwritten.
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_threshold = np.percentile(gray_roi, 10)
    dark_pixels = roi[gray_roi <= dark_threshold]
    ink = (
        tuple(int(v) for v in dark_pixels.reshape(-1, 3).mean(axis=0))
        if dark_pixels.size
        else (60, 40, 30)
    )

    # Remove the original writing. Inpainting borrows surrounding paper texture;
    # a flat median fill is the fallback if the photo module is unavailable.
    mask = np.zeros(out.shape[:2], np.uint8)
    mask[y : y + h, x : x + w] = 255
    try:
        out = cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)
    except cv2.error:  # pragma: no cover - depends on the OpenCV build
        out[y : y + h, x : x + w] = np.median(
            roi.reshape(-1, 3), axis=0
        ).astype(np.uint8)

    text = "".join(rng.choice(string.digits) for _ in range(rng.randint(6, 12)))
    scale = max(0.4, (h * 0.62) / 22.0)
    cv2.putText(
        out,
        text,
        (x + 4, y + int(h * 0.75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        ink,
        max(1, int(scale * 1.6)),
        cv2.LINE_AA,
    )
    return out


def op_requant(bgr: np.ndarray, rng: random.Random, regions) -> np.ndarray:
    """Resave one region at a much lower JPEG quality than the rest."""
    out = bgr.copy()
    x, y, w, h = _patch_rect(rng, out.shape, regions)
    roi = out[y : y + h, x : x + w]
    if roi.size == 0:
        return out

    quality = rng.randint(*_REQUANT_QUALITY)
    ok, encoded = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return out
    degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if degraded is None or degraded.shape != roi.shape:
        return out
    _blend(out, degraded, x, y, feather=2)
    return out


# --- Capture-style variation (applied to BOTH classes) ----------------------
def apply_geometric(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small rotation plus a mild perspective nudge, as if re-photographed."""
    height, width = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), rng.uniform(-2.0, 2.0), 1.0
    )
    out = cv2.warpAffine(
        bgr, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    jitter = 0.008
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = source + np.float32(
        [
            [rng.uniform(-jitter, jitter) * width, rng.uniform(-jitter, jitter) * height]
            for _ in range(4)
        ]
    )
    perspective = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        out,
        perspective,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_photometric(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness, contrast, gamma, and sensor-noise variation."""
    out = cv2.convertScaleAbs(bgr, alpha=rng.uniform(0.90, 1.10), beta=rng.uniform(-12, 12))

    gamma = rng.uniform(0.85, 1.18)
    lut = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8
    )
    out = cv2.LUT(out, lut)

    sigma = rng.uniform(0.0, 2.5)
    if sigma > 0.3:
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _limit_size(bgr: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Downscale very large scans so the dataset stays a manageable size."""
    height, width = bgr.shape[:2]
    long_edge = max(height, width)
    if max_long_edge <= 0 or long_edge <= max_long_edge:
        return bgr
    scale = max_long_edge / long_edge
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _save_jpeg(bgr: np.ndarray, path: Path, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = rng.randint(*_SAVE_QUALITY)
    # cv2.imwrite chokes on non-ASCII paths on Windows; encode then write bytes.
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.write_bytes(encoded.tobytes())


def _read(path: Path) -> np.ndarray:
    """Read an image, tolerating non-ASCII paths on Windows."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image {path}")
    return image


# --- Orchestration ----------------------------------------------------------
def _split_sources(sources: list[tuple[str, Path]], val_fraction: float, rng: random.Random):
    """Assign each *source image* to train or val (grouped split)."""
    order = sources[:]
    rng.shuffle(order)
    val_count = int(round(len(order) * val_fraction))
    # Always keep at least one source on each side when we have two or more.
    if len(order) >= 2:
        val_count = max(1, min(val_count, len(order) - 1))
    else:
        val_count = 0
    val = {path for _, path in order[:val_count]}
    return {path: ("val" if path in val else "train") for _, path in sources}


def build(
    aug_per_image: int,
    tamper_per_image: int,
    val_fraction: float,
    max_long_edge: int,
    seed: int,
    clean: bool,
) -> list[dict]:
    """Generate the dataset and return the manifest rows."""
    config.set_seed(seed)
    rng = random.Random(seed)

    sources = list(config.iter_raw_images())
    if not sources:
        raise SystemExit(
            f"No source images found under {config.RAW_DIR}.\n"
            f"Add authentic scans to {config.RAW_DIR / 'nic'} and "
            f"{config.RAW_DIR / 'proposal'} first (see ml_training/README.md)."
        )

    if clean:
        for folder in (config.GENUINE_DIR, config.TAMPERED_DIR):
            if folder.exists():
                shutil.rmtree(folder)
    config.ensure_dirs()

    splits = _split_sources(sources, val_fraction, rng)
    if len(sources) < 4:
        print(
            f"WARNING: only {len(sources)} source image(s). The scripts will run, but a "
            "classifier trained on this cannot generalize - aim for 30+ per document "
            "kind before trusting the metrics."
        )

    images = {path: _limit_size(_read(path), max_long_edge) for _, path in sources}
    rows: list[dict] = []

    for doc_kind, path in sources:
        base = images[path]
        stem = f"{doc_kind}_{path.stem}".replace(" ", "_")
        regions = find_text_regions(base)
        split = splits[path]

        for index in range(aug_per_image):
            variant = apply_photometric(apply_geometric(base, rng), rng)
            out_path = config.GENUINE_DIR / f"{stem}_aug{index:02d}.jpg"
            _save_jpeg(variant, out_path, rng)
            rows.append(
                {
                    "path": out_path.relative_to(config.ROOT_DIR).as_posix(),
                    "label": 0,
                    "source": path.stem,
                    "doc_kind": doc_kind,
                    "op": "genuine",
                    "split": split,
                }
            )

        # Donors for splicing must come from a different source image.
        donors = [images[other] for _, other in sources if other != path]

        for index in range(tamper_per_image):
            op = config.TAMPER_OPS[index % len(config.TAMPER_OPS)]
            if op == "splice" and not donors:
                # Only one source available — fall back and record it honestly.
                op = "copymove"

            if op == "copymove":
                forged = op_copymove(base, rng, regions)
            elif op == "splice":
                forged = op_splice(base, rng, regions, rng.choice(donors))
            elif op == "textpatch":
                forged = op_textpatch(base, rng, regions)
            else:
                forged = op_requant(base, rng, regions)

            # Same capture-style variation as the genuine class.
            forged = apply_photometric(apply_geometric(forged, rng), rng)
            out_path = config.TAMPERED_DIR / f"{stem}_{op}{index:02d}.jpg"
            _save_jpeg(forged, out_path, rng)
            rows.append(
                {
                    "path": out_path.relative_to(config.ROOT_DIR).as_posix(),
                    "label": 1,
                    "source": path.stem,
                    "doc_kind": doc_kind,
                    "op": op,
                    "split": split,
                }
            )

    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "source", "doc_kind", "op", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _report(rows: list[dict]) -> None:
    by_split = Counter((row["split"], "tampered" if row["label"] else "genuine") for row in rows)
    by_op = Counter(row["op"] for row in rows)

    print(f"\nWrote {len(rows)} images and {config.MANIFEST_PATH.name}")
    print(f"  manifest: {config.MANIFEST_PATH}")
    print("\n  split  class      count")
    for (split, label), count in sorted(by_split.items()):
        print(f"  {split:<6} {label:<10} {count}")
    print("\n  operation   count")
    for op, count in sorted(by_op.items()):
        print(f"  {op:<11} {count}")
    print("\nNext: python -m ml_training.train_cnn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aug-per-image", type=int, default=config.AUG_PER_IMAGE)
    parser.add_argument("--tamper-per-image", type=int, default=config.TAMPER_PER_IMAGE)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=1600,
        help="Downscale scans whose long edge exceeds this (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--clean", action="store_true", help="Delete previously generated images first."
    )
    args = parser.parse_args()

    rows = build(
        aug_per_image=args.aug_per_image,
        tamper_per_image=args.tamper_per_image,
        val_fraction=args.val_fraction,
        max_long_edge=args.max_long_edge,
        seed=args.seed,
        clean=args.clean,
    )
    _report(rows)


if __name__ == "__main__":
    main()
