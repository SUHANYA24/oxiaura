"""Score real documents through the app's own fraud pipeline (BUILD_SPEC Phase 6b).

The trainers report how each *model* scores in isolation. This reports what the
**API** returns, which is a different number: it calls
:func:`app.services.fraud_service.run_detectors` — the exact function
``POST /documents`` reaches through ``ocr_service`` — inside a real application
context, so every weight path, mock fallback and aggregate weight in effect for a
live upload is in effect here too.

That distinction is the point. A good CNN can still produce a poor verdict,
because the verdict is a weighted blend:

    aggregate = 0.3*ELA + 0.4*CNN*100 + 0.3*Siamese*100

Any detector still on its deterministic mock contributes a hash-derived number in
[0, 1] to that sum — stable per file, but unrelated to whether the document was
tampered with. This script measures each term's ROC AUC separately, so a mock
term shows up as ~0.5 and you can see exactly how much it costs the aggregate.

Scored on the **held-out ``val`` split** of ``datasets/fraud/manifest.csv``. The
split is grouped by source scan, so no document here contributed a variant to
CNN training and the numbers are not inflated by memorization.

Reads the dataset and the app config; writes one JSON report. Changes nothing.

Usage::

    python -m ml_training.evaluate_pipeline
    python -m ml_training.evaluate_pipeline --limit 0        # every val image
    python -m ml_training.evaluate_pipeline --limit 60       # quick look
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from . import config
from .train_cnn import load_manifest, roc_auc

REPORT_PATH = config.DATASETS_DIR / "pipeline_report.json"


def detector_status(app) -> dict[str, dict]:
    """Describe which detectors will run for real, by the same tests they use.

    Mirrors the conditions inside each detector rather than trusting the config:
    ``predict_tamper_probability`` requires a readable weights file, and
    ``highest_similarity`` requires weights *and* a non-empty bank.
    """
    cnn_weights = app.config.get("FRAUD_CNN_WEIGHTS")
    siamese_weights = app.config.get("FRAUD_SIAMESE_WEIGHTS")
    bank_path = app.config.get("FRAUD_REFERENCE_BANK")

    bank_count = 0
    if bank_path and Path(bank_path).is_file():
        import torch

        bank = torch.load(bank_path, map_location="cpu")
        bank_count = 0 if bank is None else len(bank)

    return {
        "ela": {
            "real": True,
            "why": "deterministic, needs no weights",
        },
        "cnn": {
            "real": bool(cnn_weights and Path(cnn_weights).is_file()),
            "why": (
                f"weights at {cnn_weights}"
                if cnn_weights and Path(cnn_weights).is_file()
                else f"FRAUD_CNN_WEIGHTS={cnn_weights!r} is unset or missing"
            ),
        },
        "siamese": {
            "real": bool(
                siamese_weights
                and Path(siamese_weights).is_file()
                and bank_count > 0
            ),
            "why": (
                f"{bank_count} exemplar(s) in the bank"
                if bank_count
                else "no reference bank, so nothing to compare an upload against"
            ),
        },
    }


def pick_rows(limit: int, seed: int) -> list[dict]:
    """Take a label-balanced sample of the val split, or all of it if limit is 0."""
    rows = load_manifest("val")
    if not rows:
        raise SystemExit(
            "The manifest has no 'val' rows. Run: python -m ml_training.prepare_dataset"
        )
    if limit <= 0 or limit >= len(rows):
        return rows

    # Balanced by hand rather than by a plain sample: an unbalanced draw would
    # move the AUCs around for a reason that has nothing to do with the models.
    genuine = [row for row in rows if row["label"] == 0]
    tampered = [row for row in rows if row["label"] == 1]
    rng = random.Random(seed)
    rng.shuffle(genuine)
    rng.shuffle(tampered)
    half = limit // 2
    return genuine[:half] + tampered[: limit - half]


def score_rows(rows: list[dict], run_detectors) -> list[dict]:
    """Run the real pipeline over every row, printing progress as it goes."""
    results = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        path = config.ROOT_DIR / row["path"]
        if not path.is_file():
            print(f"  missing, skipped: {row['path']}", flush=True)
            continue
        scores = run_detectors(str(path))
        results.append(
            {
                "path": row["path"],
                "label": row["label"],
                "doc_kind": row["doc_kind"],
                "op": row["op"],
                "ela": scores["ela_score"],
                "cnn": scores["cnn_fraud_score"],
                "siamese": scores["siamese_similarity"],
                "aggregate": scores["aggregate_score"],
                "is_flagged": scores["is_flagged"],
            }
        )
        if index % 25 == 0 or index == len(rows):
            rate = (time.time() - started) / index
            print(
                f"  {index}/{len(rows)} scored  ({rate:.2f}s per image)",
                flush=True,
            )
    if not results:
        raise SystemExit("Nothing could be scored - every image path was missing.")
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _by_class(results: list[dict], key: str) -> tuple[float, float]:
    return (
        _mean([r[key] for r in results if r["label"] == 0]),
        _mean([r[key] for r in results if r["label"] == 1]),
    )


def _counterfactual(result: dict) -> float:
    """The aggregate with the Siamese term dropped and the rest renormalized.

    Answers the question the mock raises: what would the verdict look like if the
    unavailable detector simply did not vote, instead of voting at random?
    """
    return (0.3 * result["ela"] + 0.4 * result["cnn"] * 100) / 0.7


def _confusion(results: list[dict], threshold: float, key: str) -> dict:
    tp = sum(1 for r in results if r["label"] == 1 and r[key] > threshold)
    fp = sum(1 for r in results if r["label"] == 0 and r[key] > threshold)
    tn = sum(1 for r in results if r["label"] == 0 and r[key] <= threshold)
    fn = sum(1 for r in results if r["label"] == 1 and r[key] <= threshold)
    total = tp + fp + tn + fn
    return {
        "threshold": round(threshold, 2),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": round((tp + tn) / total, 4) if total else float("nan"),
        "precision": round(tp / (tp + fp), 4) if tp + fp else float("nan"),
        "recall": round(tp / (tp + fn), 4) if tp + fn else float("nan"),
    }


def _best_threshold(results: list[dict], key: str) -> tuple[float, float]:
    """The cut-off on ``key`` with the highest accuracy, and that accuracy."""
    labels = [r["label"] for r in results]
    candidates = sorted({round(r[key], 2) for r in results})
    best, best_accuracy = float("nan"), -1.0
    for threshold in candidates:
        correct = sum(
            1
            for label, r in zip(labels, results)
            if (r[key] > threshold) == (label == 1)
        )
        accuracy = correct / len(results)
        if accuracy > best_accuracy:
            best, best_accuracy = threshold, accuracy
    return best, best_accuracy


def summarize(results: list[dict], status: dict, threshold: float) -> dict:
    """Print the report and return it as a JSON-serializable dict."""
    labels = [float(r["label"]) for r in results]
    for result in results:
        result["aggregate_without_siamese"] = round(_counterfactual(result), 2)

    genuine = sum(1 for r in results if r["label"] == 0)
    tampered = len(results) - genuine

    print(f"\nScored {len(results)} held-out images: {genuine} genuine, {tampered} tampered")
    print("(val split only - grouped by source, so the CNN never trained on these)\n")

    print("Detectors in effect for these scores:")
    for name in ("ela", "cnn", "siamese"):
        state = "REAL" if status[name]["real"] else "MOCK"
        print(f"  {name:8} {state:5} - {status[name]['why']}")

    signals = {
        "ela": ("ELA (0-100)", "ela"),
        "cnn": ("CNN P(tampered)", "cnn"),
        "siamese": ("Siamese similarity", "siamese"),
        "aggregate": ("AGGREGATE (0-100)", "aggregate"),
        "aggregate_without_siamese": (
            "aggregate, no Siamese",
            "aggregate_without_siamese",
        ),
    }

    print("\nPer-signal separation (higher AUC = better at telling the classes apart):")
    print(f"  {'signal':24} {'genuine':>9} {'tampered':>9} {'gap':>8} {'AUC':>7}")
    aucs = {}
    for key, (label, field) in signals.items():
        clean, dirty = _by_class(results, field)
        auc = roc_auc(labels, [r[field] for r in results])
        aucs[key] = auc
        print(
            f"  {label:24} {clean:9.3f} {dirty:9.3f} {dirty - clean:+8.3f} {auc:7.3f}"
        )
    print("  AUC 0.5 = no signal at all; a detector on its mock lands there.")

    print(f"\nVerdict at the configured FRAUD_FLAG_THRESHOLD={threshold:g}:")
    live = _confusion(results, threshold, "aggregate")
    print(
        f"  flagged {live['true_positives'] + live['false_positives']} of {len(results)}"
        f"  |  accuracy {live['accuracy']:.3f}"
        f"  precision {live['precision']:.3f}  recall {live['recall']:.3f}"
    )
    print(
        f"  caught {live['true_positives']}/{tampered} tampered, "
        f"missed {live['false_negatives']}; "
        f"false-flagged {live['false_positives']}/{genuine} genuine"
    )

    best, best_accuracy = _best_threshold(results, "aggregate")
    print(
        f"\n  Best aggregate cut-off on this sample: {best:.2f} "
        f"(accuracy {best_accuracy:.3f} vs {live['accuracy']:.3f} at {threshold:g})"
    )
    alt_best, alt_accuracy = _best_threshold(results, "aggregate_without_siamese")
    print(
        f"  Dropping the Siamese term and renormalizing: best cut-off "
        f"{alt_best:.2f}, accuracy {alt_accuracy:.3f}"
    )

    print("\nBy document kind (aggregate AUC):")
    for kind in sorted({r["doc_kind"] for r in results}):
        subset = [r for r in results if r["doc_kind"] == kind]
        sub_auc = roc_auc(
            [float(r["label"]) for r in subset], [r["aggregate"] for r in subset]
        )
        print(f"  {kind:10} n={len(subset):4}  AUC {sub_auc:.3f}")

    print("\nBy tamper operation (recall at the configured threshold):")
    for op in sorted({r["op"] for r in results if r["label"] == 1}):
        subset = [r for r in results if r["op"] == op]
        caught = sum(1 for r in subset if r["aggregate"] > threshold)
        print(f"  {op:12} n={len(subset):4}  caught {caught:4}  ({caught / len(subset):.0%})")

    return {
        "scored": len(results),
        "genuine": genuine,
        "tampered": tampered,
        "detectors": status,
        "flag_threshold": threshold,
        "auc": {key: (None if auc != auc else round(auc, 4)) for key, auc in aucs.items()},
        "confusion_at_configured_threshold": live,
        "best_threshold": {
            "aggregate": round(best, 2),
            "accuracy": round(best_accuracy, 4),
            "aggregate_without_siamese": round(alt_best, 2),
            "accuracy_without_siamese": round(alt_accuracy, 4),
        },
        "per_image": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Balanced sample size from the val split; 0 scores every val image.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--config",
        default="development",
        help="Flask config name, so you can score against production settings.",
    )
    args = parser.parse_args()

    # Imported here, not at module scope: this pulls in the whole Flask app.
    from app import create_app
    from app.services import fraud_service

    app = create_app(args.config)
    rows = pick_rows(args.limit, args.seed)
    print(f"Scoring {len(rows)} image(s) through app.services.fraud_service.run_detectors")

    with app.app_context():
        status = detector_status(app)
        results = score_rows(rows, fraud_service.run_detectors)
        threshold = float(app.config.get("FRAUD_FLAG_THRESHOLD", 60))
        report = summarize(results, status, threshold)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPer-image detail: {REPORT_PATH}")

    if not report["detectors"]["siamese"]["real"]:
        print(
            "\nNote: the Siamese term is a mock, so 30% of every aggregate above is\n"
            "a hash-derived number unrelated to tampering. Compare the two\n"
            "'aggregate' rows to see what that costs."
        )


if __name__ == "__main__":
    main()
