"""Scaffold OCR ground-truth label files (BUILD_SPEC Phase 6b).

``evaluate_ocr.py`` needs to know what the *correct* answer is for each scan
before it can tell you how well extraction is doing. That part cannot be
automated — it is the manual dataset you have to create. This script removes the
tedium: it writes one JSON stub per raw scan, pre-filled with the field keys
that document actually has, and you type in the values.

Field sets come from the real documents:

* **NIC front** — NIC number, name, sex, date of birth
* **NIC back** — address, date of issue, place of birth
* **Proposal front** — customer and nominee blocks
* **Proposal back** — product/deposit, bank details, witness blocks

Front and back are told apart by the filename, so keep ``front``/``back`` in
your scan names (``proposal-2-front.jpeg``).

An existing stub is never overwritten unless you pass ``--force``, so re-running
this after adding new scans is safe — your typed-in labels are not at risk.

Usage::

    python -m ml_training.label_ocr                 # create missing stubs
    python -m ml_training.label_ocr --status        # what still needs filling in
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config

# Keys per document kind and side, in reading order.
FIELD_SETS: dict[tuple[str, str], tuple[str, ...]] = {
    ("nic", "front"): (
        "nic_number",
        "name",
        "sex",
        "date_of_birth",
    ),
    ("nic", "back"): (
        "address",
        "date_of_issue",
        "place_of_birth",
    ),
    ("proposal", "front"): (
        "customer_full_name",
        "customer_name_with_initials",
        "customer_address",
        "customer_nic",
        "customer_birthday",
        "customer_tel_phone",
        "customer_email",
        "nominee_full_name",
        "nominee_name_with_initials",
        "nominee_address",
        "nominee_nic",
        "nominee_birthday",
        "nominee_relationship",
        "nominee_tel",
    ),
    ("proposal", "back"): (
        "proposal_no",
        "agreement_no",
        "product",
        "mode_of_payment",
        "deposited",
        "period_months",
        "account_number",
        "account_holder_name",
        "bank",
        "branch",
        "signature_date",
        "employee_name",
        "employee_code",
        "employee_nic",
        "supervisor_name",
        "supervisor_code",
        "supervisor_nic",
    ),
}


def detect_side(path: Path) -> str:
    """Infer ``front``/``back`` from the filename, defaulting to ``front``."""
    stem = path.stem.lower()
    if "back" in stem:
        return "back"
    if "front" in stem:
        return "front"
    return "front"


def field_keys(doc_kind: str, side: str) -> tuple[str, ...]:
    """Field keys for a document kind/side, or the union when kind is unknown."""
    if (doc_kind, side) in FIELD_SETS:
        return FIELD_SETS[(doc_kind, side)]
    # Unrecognized kind: offer everything for that side so nothing is lost.
    keys: list[str] = []
    for (_, known_side), values in FIELD_SETS.items():
        if known_side == side:
            keys.extend(values)
    return tuple(dict.fromkeys(keys))


def label_path(doc_kind: str, image_path: Path) -> Path:
    """Where the label file for a given scan lives."""
    return config.OCR_LABELS_DIR / f"{doc_kind}__{image_path.stem}.json"


def make_stub(doc_kind: str, image_path: Path) -> dict:
    """Build an empty label document for one scan."""
    side = detect_side(image_path)
    relative = (
        image_path.relative_to(config.ROOT_DIR).as_posix()
        if image_path.is_relative_to(config.ROOT_DIR)
        else image_path.as_posix()
    )
    return {
        "image": relative,
        "doc_kind": doc_kind,
        "side": side,
        # Leave a value empty when the field is genuinely blank on the scan;
        # evaluate_ocr.py skips empty keys rather than counting them as misses.
        "fields": {key: "" for key in field_keys(doc_kind, side)},
    }


def scaffold(force: bool, doc_kinds) -> tuple[int, int]:
    """Write missing stubs. Returns ``(created, skipped)``."""
    config.ensure_dirs()
    created, skipped = 0, 0

    for doc_kind, image_path in config.iter_raw_images():
        if doc_kinds and doc_kind not in doc_kinds:
            continue
        target = label_path(doc_kind, image_path)
        if target.exists() and not force:
            skipped += 1
            continue
        target.write_text(
            json.dumps(make_stub(doc_kind, image_path), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  wrote {target.relative_to(config.ROOT_DIR).as_posix()}")
        created += 1

    return created, skipped


def status() -> None:
    """Report how much of each label file is still empty."""
    files = sorted(config.OCR_LABELS_DIR.glob("*.json"))
    if not files:
        print(f"No label files in {config.OCR_LABELS_DIR}. Run without --status first.")
        return

    print(f"{'label file':<44} {'filled':>8} {'empty':>7}")
    total_filled = total_empty = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = data.get("fields", {})
        filled = sum(1 for value in fields.values() if str(value).strip())
        empty = len(fields) - filled
        total_filled += filled
        total_empty += empty
        print(f"{path.name:<44} {filled:>8} {empty:>7}")

    print(f"{'TOTAL':<44} {total_filled:>8} {total_empty:>7}")
    if total_filled == 0:
        print(
            "\nNothing filled in yet. Open the JSON files next to the scans and type the "
            "values you can read; leave a field empty if it is blank on the document."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--doc-kind",
        action="append",
        dest="doc_kinds",
        help="Limit to one document kind (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing label files - DISCARDS values you have typed in.",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show how many fields are still empty."
    )
    args = parser.parse_args()

    if args.status:
        status()
        return

    created, skipped = scaffold(args.force, args.doc_kinds)
    print(f"\nCreated {created} stub(s), left {skipped} existing file(s) untouched.")
    print(f"Labels live in: {config.OCR_LABELS_DIR}")
    if created:
        print(
            "Fill in the values by hand, then run:\n"
            "  python -m ml_training.label_ocr --status\n"
            "  python -m ml_training.evaluate_ocr"
        )


if __name__ == "__main__":
    main()
