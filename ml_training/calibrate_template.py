"""Calibrate a field-box template for a fixed-layout document (Phase 6b).

``app/ai/ocr/template_extractor.py`` needs a JSON file saying where each field
sits on a reference scan. This script helps you produce one without guessing
coordinates in the dark.

The project pins ``opencv-python-headless``, which ships **no GUI**, so there is
no window to drag boxes in. Instead this works by generating images you look at
and a JSON you edit:

``--auto``
    Detect the ruled table structure on the scan and emit a starter template
    with one box per detected cell, named by row/column (``field_r03c02``).
    Good on the proposal form; the NIC card has no ruled table, so use
    ``--grid`` for that one.

``--grid``
    Render the scan with a labelled 5%% grid on top, so you can read normalized
    coordinates straight off the picture and type them into the JSON.

``--preview``
    Render the template's current boxes, labelled, onto the reference scan — the
    check that your edits landed where you meant.

Front and back are different layouts, so each gets its own template. The output
is named ``<doc-kind>_<side>.json`` (``proposal_back.json``), which is what
``evaluate_ocr.py`` looks for.

Typical loop::

    python -m ml_training.calibrate_template --doc-kind proposal --side back --auto
    python -m ml_training.calibrate_template --doc-kind proposal --side back --preview
    # rename field_r03c02 -> "deposited" in the JSON, delete boxes you do not need
    python -m ml_training.calibrate_template --doc-kind proposal --side back --preview

For the NIC card, which is usually photographed sideways — pass ``--rotate`` so
you read coordinates off an upright picture (the rotation is recorded in the
template and re-applied at OCR time; the scan file itself is never touched)::

    python -m ml_training.calibrate_template --doc-kind nic --side front \
        --rotate 90 --grid
    # hand-write the boxes into ml_training/templates/nic_front.json, then --preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.ai.ocr import template_extractor

from . import config, label_ocr

# Cells smaller than this fraction of the page are noise; larger ones are the
# table outline rather than a cell.
_MIN_CELL_AREA = 0.0008
_MAX_CELL_AREA = 0.30


def _write_png(image: np.ndarray, path: Path) -> Path:
    """Write a PNG, tolerating non-ASCII paths on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.write_bytes(encoded.tobytes())
    return path


def detect_cells(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find ruled table cells and return them as ``(x, y, w, h)`` pixel rects.

    Long horizontal and vertical strokes are isolated with directional
    morphology; their union is the table skeleton, and its enclosed contours are
    the cells. This is a starting point for hand-editing, not a solved layout —
    a faint or broken rule line will merge two cells.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    height, width = gray.shape[:2]

    binary = cv2.adaptiveThreshold(
        cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, width // 40), 1))
    horizontal = cv2.dilate(cv2.erode(binary, h_kernel), h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 40)))
    vertical = cv2.dilate(cv2.erode(binary, v_kernel), v_kernel)

    skeleton = cv2.dilate(cv2.add(horizontal, vertical), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    page_area = float(width * height)
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = (w * h) / page_area
        if _MIN_CELL_AREA <= area <= _MAX_CELL_AREA and w > 30 and h > 14:
            cells.append((x, y, w, h))
    return cells


def group_into_rows(cells: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Sort cells into visual rows, each ordered left to right."""
    if not cells:
        return []

    by_top = sorted(cells, key=lambda cell: cell[1])
    tolerance = max(8, int(np.median([cell[3] for cell in by_top]) * 0.6))

    rows: list[list[tuple[int, int, int, int]]] = [[by_top[0]]]
    for cell in by_top[1:]:
        if abs(cell[1] - rows[-1][0][1]) <= tolerance:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    for row in rows:
        row.sort(key=lambda cell: cell[0])
    return rows


def build_template(
    image: np.ndarray, name: str, doc_kind: str, side: str, reference_image: str, rotate: int = 0
) -> dict:
    """Turn detected cells into a starter template dict."""
    height, width = image.shape[:2]
    rows = group_into_rows(detect_cells(image))

    fields: dict[str, dict] = {}
    for row_index, row in enumerate(rows, start=1):
        for col_index, (x, y, w, h) in enumerate(row, start=1):
            fields[f"field_r{row_index:02d}c{col_index:02d}"] = {
                "box": [
                    round(x / width, 4),
                    round(y / height, 4),
                    round(w / width, 4),
                    round(h / height, 4),
                ]
            }

    return {
        "name": name,
        "doc_kind": doc_kind,
        "side": side,
        "reference_image": reference_image,
        "reference_size": [width, height],
        "rotate": rotate,
        "fields": fields,
    }


def render_boxes(image: np.ndarray, template: dict) -> np.ndarray:
    """Draw every templated box, labelled, onto a copy of ``image``."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.4, min(width, height) / 1400.0)

    for index, (name, spec) in enumerate(template["fields"].items()):
        x, y, w, h = spec["box"]
        x0, y0 = int(x * width), int(y * height)
        x1, y1 = int((x + w) * width), int((y + h) * height)
        # Cycle three colours so adjacent boxes stay tellable apart.
        colour = [(0, 170, 255), (0, 220, 0), (255, 80, 200)][index % 3]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(
            canvas,
            name,
            (x0 + 3, max(14, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.55,
            colour,
            max(1, int(scale * 1.4)),
            cv2.LINE_AA,
        )
    return canvas


def render_grid(image: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Overlay a labelled percentage grid for reading coordinates by eye."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.35, min(width, height) / 1600.0)

    steps = int(round(1.0 / step))
    for index in range(steps + 1):
        fraction = index * step
        x, y = int(fraction * width), int(fraction * height)
        # Every other line is heavier, so counting is easier.
        thickness = 2 if index % 2 == 0 else 1
        colour = (0, 0, 255) if index % 2 == 0 else (180, 180, 180)

        cv2.line(canvas, (x, 0), (x, height), colour, thickness)
        cv2.line(canvas, (0, y), (width, y), colour, thickness)

        if index % 2 == 0:
            label = f"{fraction:.2f}"
            cv2.putText(
                canvas, label, (min(x + 4, width - 40), 18),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, (0, 0, 255),
                max(1, int(scale * 1.3)), cv2.LINE_AA,
            )
            cv2.putText(
                canvas, label, (4, min(y + 16, height - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, (0, 0, 255),
                max(1, int(scale * 1.3)), cv2.LINE_AA,
            )
    return canvas


def pick_reference(doc_kind: str, side: str | None, explicit: Path | None) -> Path:
    """Resolve the reference scan for ``doc_kind`` (optionally a given side)."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"No such image: {explicit}")
        return explicit

    candidates = [path for kind, path in config.iter_raw_images() if kind == doc_kind]
    if not candidates:
        raise SystemExit(
            f"No images found in {config.RAW_DIR / doc_kind}.\n"
            "Add the scan you want to calibrate against, or pass --image."
        )

    if side:
        matching = [path for path in candidates if label_ocr.detect_side(path) == side]
        if not matching:
            raise SystemExit(
                f"No '{side}' scan among {[p.name for p in candidates]}.\n"
                f"Put '{side}' in the filename, or pass --image explicitly."
            )
        candidates = matching
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--doc-kind", default="proposal", choices=list(config.DOC_KINDS) + ["unknown"]
    )
    parser.add_argument("--image", type=Path, help="Reference scan (defaults to the first found).")
    parser.add_argument(
        "--side",
        choices=("front", "back"),
        help="Which page to calibrate. Front and back are different layouts and need "
        "separate templates; omit only if the kind has a single page.",
    )
    parser.add_argument(
        "--name", help="Template name (defaults to '<doc-kind>_<side>', e.g. proposal_back)."
    )
    parser.add_argument(
        "--auto", action="store_true", help="Detect table cells and write a starter template."
    )
    parser.add_argument(
        "--grid", action="store_true", help="Write a coordinate-grid overlay PNG."
    )
    parser.add_argument(
        "--preview", action="store_true", help="Write a PNG of the template's current boxes."
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=None,
        choices=(0, 90, 180, 270),
        help="Counter-clockwise degrees to straighten a sideways scan (an ID card "
        "photographed landscape in a portrait frame). Applied before everything else, "
        "so you read coordinates the way you read the document. Defaults to whatever "
        "the existing template records, else 0.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Let --auto overwrite an existing template."
    )
    args = parser.parse_args()

    if not (args.auto or args.grid or args.preview):
        parser.error("Choose at least one of --auto, --grid, --preview.")

    config.ensure_dirs()
    reference_path = pick_reference(args.doc_kind, args.side, args.image)
    # The side comes from the chosen scan, so --image alone still names correctly.
    side = args.side or label_ocr.detect_side(reference_path)
    # Front and back are separate layouts, so the side belongs in the filename —
    # evaluate_ocr.py looks for '<kind>_<side>.json'.
    name = args.name or f"{args.doc_kind}_{side}"
    template_path = config.TEMPLATES_DIR / f"{name}.json"

    # Carry the rotation the template already records, so a --preview after a
    # hand-edit renders in the same frame the boxes were written in.
    rotate = args.rotate
    if rotate is None:
        rotate = 0
        if template_path.is_file():
            rotate = int(template_extractor.load_template(template_path).get("rotate", 0))

    image = template_extractor.rotate_image(
        template_extractor.read_image(reference_path), rotate
    )

    print(
        f"reference: {reference_path}  ({image.shape[1]}x{image.shape[0]})  "
        f"side={side}  rotate={rotate}"
    )

    if args.auto:
        if template_path.exists() and not args.force:
            raise SystemExit(
                f"{template_path} already exists - pass --force to overwrite it, or "
                "use --preview to inspect what is there."
            )
        relative = (
            reference_path.relative_to(config.ROOT_DIR).as_posix()
            if reference_path.is_relative_to(config.ROOT_DIR)
            else reference_path.as_posix()
        )
        template = build_template(image, name, args.doc_kind, side, relative, rotate)
        template_extractor.save_template(template, template_path)
        print(f"detected {len(template['fields'])} cells -> {template_path}")
        print(
            "  Now rename the useful boxes to real field names (nic, full_name, ...) "
            "and delete the rest."
        )

    if args.grid:
        path = _write_png(render_grid(image), config.TEMPLATES_DIR / f"{name}.grid.png")
        print(f"grid overlay -> {path}")

    if args.preview:
        template = template_extractor.load_template(template_path)
        path = _write_png(
            render_boxes(image, template), config.TEMPLATES_DIR / f"{name}.preview.png"
        )
        print(f"box preview ({len(template['fields'])} fields) -> {path}")


if __name__ == "__main__":
    main()
