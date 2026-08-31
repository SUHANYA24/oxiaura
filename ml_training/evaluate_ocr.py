"""Measure OCR field-extraction accuracy against hand-written labels (Phase 6b).

Runs the real extraction pipeline over your labelled scans and reports, per
field, how often a value was found and how close it was to the truth. Two
strategies are scored so you can see whether the template pays for itself:

``regex``
    The current production path — ``preprocessor.preprocess`` over the whole
    page, then ``extractor.parse_fields``' regex patterns.

``template``
    ``app/ai/ocr/template_extractor.py`` — align to the reference scan, then OCR
    each field box on its own.

Metrics per field:

``found``       a non-empty value was produced
``exact``       character-for-character match
``normalized``  match ignoring case, spacing, and punctuation (``1990-06-18``
                and ``1990 06 18`` count as equal)
``CER``         character error rate, edit distance / truth length — the useful
                one when a value is *nearly* right

The regex extractor names fields generically (``nic_number``, ``name``,
``date``), while labels are specific (``customer_nic``, ``nominee_birthday``).
:data:`FIELD_ALIASES` maps between them so both strategies are scored against
the same ground truth. Template fields are matched by name directly — which is
why the box names in your template should match the label keys.

This loads EasyOCR and runs it over every field of every scan, so it is slow on
CPU (and the very first run downloads the recognition models). Use ``--limit``
while iterating.

Usage::

    python -m ml_training.evaluate_ocr
    python -m ml_training.evaluate_ocr --mode template --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.ai.ocr import extractor, preprocessor, template_extractor

from . import config

# Label key -> the generic keys the regex extractor may produce for it.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "nic_number": ("nic_number",),
    "customer_nic": ("nic_number",),
    "nominee_nic": ("nic_number",),
    "employee_nic": ("nic_number",),
    "supervisor_nic": ("nic_number",),
    "name": ("name",),
    "customer_full_name": ("name",),
    "nominee_full_name": ("name",),
    "account_holder_name": ("name",),
    "employee_name": ("name",),
    "supervisor_name": ("name",),
    "address": ("address",),
    "customer_address": ("address",),
    "nominee_address": ("address",),
    "date_of_birth": ("date",),
    "date_of_issue": ("date",),
    "customer_birthday": ("date",),
    "nominee_birthday": ("date",),
    "signature_date": ("date",),
    "deposited": ("amount",),
    "account_number": ("account_no",),
}

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize(value: str) -> str:
    """Lowercase and strip everything that is not a letter or digit."""
    return _NON_ALNUM.sub("", str(value).lower())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (two-row DP — no external dependency)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(truth: str, predicted: str) -> float:
    """Edit distance normalized by the truth length; 1.0 when nothing matched."""
    truth_norm, predicted_norm = normalize(truth), normalize(predicted)
    if not truth_norm:
        return 0.0
    return min(1.0, edit_distance(truth_norm, predicted_norm) / len(truth_norm))


def load_labels(doc_kinds=None, limit: int | None = None) -> list[dict]:
    """Load label files that have at least one value filled in."""
    if not config.OCR_LABELS_DIR.is_dir():
        raise SystemExit(
            f"No labels directory at {config.OCR_LABELS_DIR}.\n"
            "Run: python -m ml_training.label_ocr"
        )

    labels, empty = [], 0
    for path in sorted(config.OCR_LABELS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        filled = {
            key: str(value).strip()
            for key, value in data.get("fields", {}).items()
            if str(value).strip()
        }
        if not filled:
            empty += 1
            continue
        if doc_kinds and data.get("doc_kind") not in doc_kinds:
            continue

        image_path = config.ROOT_DIR / data["image"]
        if not image_path.is_file():
            print(f"  skipping {path.name}: image not found at {image_path}")
            continue

        data["fields"] = filled
        data["image_path"] = image_path
        data["label_file"] = path.name
        labels.append(data)

    if empty:
        print(f"Skipped {empty} label file(s) with nothing filled in yet.")
    if not labels:
        raise SystemExit(
            "No usable labels. Fill in values with your editor, then check with:\n"
            "  python -m ml_training.label_ocr --status"
        )
    return labels[:limit] if limit else labels


def run_regex(image_path: Path) -> dict:
    """The current production path, whole-page OCR plus regex parsing."""
    image = preprocessor.preprocess(str(image_path))
    return extractor.parse_fields(extractor.run_ocr(image))


def run_template(image_path: Path, template: dict) -> dict:
    """The ROI path: align to the reference, OCR each field box."""
    return template_extractor.extract_from_path(image_path, template)


def score_document(predicted: dict, truth: dict[str, str]) -> list[dict]:
    """Compare one document's predictions against its labels, field by field."""
    fields = predicted.get("fields", {})
    results = []

    for key, expected in truth.items():
        # Prefer an exact key match (template mode); fall back to the generic
        # names the regex extractor emits.
        candidates = (key,) + FIELD_ALIASES.get(key, ())
        got = next((fields[name]["value"] for name in candidates if name in fields), None)

        results.append(
            {
                "field": key,
                "expected": expected,
                "got": got,
                "found": got is not None,
                "exact": got == expected,
                "normalized": got is not None and normalize(got) == normalize(expected),
                "cer": character_error_rate(expected, got or ""),
            }
        )
    return results


def aggregate(rows: list[dict]) -> dict:
    """Roll per-field results up into per-field and overall summaries."""
    by_field: dict[str, dict] = {}
    for row in rows:
        bucket = by_field.setdefault(
            row["field"], {"n": 0, "found": 0, "exact": 0, "normalized": 0, "cer": 0.0}
        )
        bucket["n"] += 1
        bucket["found"] += int(row["found"])
        bucket["exact"] += int(row["exact"])
        bucket["normalized"] += int(row["normalized"])
        bucket["cer"] += row["cer"]

    for bucket in by_field.values():
        bucket["found_rate"] = round(bucket["found"] / bucket["n"], 4)
        bucket["exact_rate"] = round(bucket["exact"] / bucket["n"], 4)
        bucket["normalized_rate"] = round(bucket["normalized"] / bucket["n"], 4)
        bucket["mean_cer"] = round(bucket["cer"] / bucket["n"], 4)

    total = len(rows) or 1
    overall = {
        "fields_evaluated": len(rows),
        "found_rate": round(sum(r["found"] for r in rows) / total, 4),
        "exact_rate": round(sum(r["exact"] for r in rows) / total, 4),
        "normalized_rate": round(sum(r["normalized"] for r in rows) / total, 4),
        "mean_cer": round(sum(r["cer"] for r in rows) / total, 4),
    }
    return {"by_field": by_field, "overall": overall}


def _print_table(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'field':<30} {'n':>3} {'found':>7} {'exact':>7} {'norm':>7} {'CER':>6}")
    for field, bucket in sorted(summary["by_field"].items()):
        print(
            f"{field:<30} {bucket['n']:>3} {bucket['found_rate']:>7.2f} "
            f"{bucket['exact_rate']:>7.2f} {bucket['normalized_rate']:>7.2f} "
            f"{bucket['mean_cer']:>6.2f}"
        )
    overall = summary["overall"]
    print(
        f"{'OVERALL':<30} {overall['fields_evaluated']:>3} {overall['found_rate']:>7.2f} "
        f"{overall['exact_rate']:>7.2f} {overall['normalized_rate']:>7.2f} "
        f"{overall['mean_cer']:>6.2f}"
    )


def find_template(doc_kind: str, side: str):
    """Locate the template for a kind/side, preferring the side-specific file.

    Front and back are different layouts, so ``proposal_back.json`` must not be
    used to score a front scan. A plain ``<kind>.json`` is accepted as a fallback
    for single-page documents.
    """
    for candidate in (
        config.TEMPLATES_DIR / f"{doc_kind}_{side}.json",
        config.TEMPLATES_DIR / f"{doc_kind}.json",
    ):
        if candidate.is_file():
            return template_extractor.load_template(candidate), candidate
    return None, config.TEMPLATES_DIR / f"{doc_kind}_{side}.json"


def evaluate(mode: str, doc_kinds, limit: int | None) -> dict:
    labels = load_labels(doc_kinds, limit)
    modes = ("regex", "template") if mode == "both" else (mode,)

    # Keyed by (doc_kind, side) — see find_template.
    templates: dict[tuple[str, str], dict] = {}
    if "template" in modes:
        for key in {(label["doc_kind"], label["side"]) for label in labels}:
            template, path = find_template(*key)
            if template is None:
                print(
                    f"  no template for {key[0]}/{key[1]} (looked for {path.name}) - those "
                    "scans are skipped in template mode. Create one with:\n"
                    f"    python -m ml_training.calibrate_template --doc-kind {key[0]} "
                    f"--side {key[1]} --auto --preview"
                )
                continue
            templates[key] = template

    results: dict[str, list[dict]] = {name: [] for name in modes}
    details: list[dict] = []

    for label in labels:
        print(f"\n{label['label_file']}  ({label['doc_kind']}/{label['side']})")
        entry = {
            "label_file": label["label_file"],
            "doc_kind": label["doc_kind"],
            "side": label["side"],
        }

        for name in modes:
            if name == "regex":
                predicted = run_regex(label["image_path"])
            else:
                template = templates.get((label["doc_kind"], label["side"]))
                if template is None:
                    continue
                predicted = run_template(label["image_path"], template)
                alignment = predicted.get("alignment", {})
                print(
                    f"  alignment: {'ok' if alignment.get('aligned') else 'FALLBACK'} "
                    f"(matches={alignment.get('matches')}, inliers={alignment.get('inliers')})"
                )

            scored = score_document(predicted, label["fields"])
            results[name].extend(scored)
            entry[name] = {
                "mean_confidence": predicted.get("mean_confidence"),
                "fields": scored,
            }
            exact = sum(1 for row in scored if row["exact"])
            near = sum(1 for row in scored if row["normalized"])
            print(f"  {name:<9} exact {exact}/{len(scored)}   normalized {near}/{len(scored)}")

        details.append(entry)

    summaries = {name: aggregate(rows) for name, rows in results.items() if rows}
    for name, summary in summaries.items():
        _print_table(name, summary)

    if len(summaries) == 2:
        regex_rate = summaries["regex"]["overall"]["normalized_rate"]
        template_rate = summaries["template"]["overall"]["normalized_rate"]
        verdict = (
            "the template wins"
            if template_rate > regex_rate
            else "the regex path wins" if regex_rate > template_rate else "a tie"
        )
        print(
            f"\nnormalized accuracy - regex {regex_rate:.2f} vs template "
            f"{template_rate:.2f}: {verdict}"
        )

    report = {
        "documents": len(labels),
        "modes": list(summaries),
        "summaries": summaries,
        "details": details,
    }
    config.OCR_DIR.mkdir(parents=True, exist_ok=True)
    config.OCR_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("regex", "template", "both"), default="both")
    parser.add_argument(
        "--doc-kind", action="append", dest="doc_kinds", help="Limit to a kind (repeatable)."
    )
    parser.add_argument("--limit", type=int, help="Evaluate at most this many documents.")
    args = parser.parse_args()

    evaluate(args.mode, args.doc_kinds, args.limit)
    print(f"\nFull per-field report: {config.OCR_REPORT_PATH}")


if __name__ == "__main__":
    main()
