"""Chart what the trainers measured (BUILD_SPEC Phase 6b).

``train_cnn`` and ``train_siamese`` already write every epoch to a sibling
``.metrics.json``, and ``prepare_dataset`` writes ``manifest.csv``. This script
only reads those files and draws them — it never loads a ``.pt``, never runs a
model, and never touches a raw scan. So it is cheap to re-run, and safe to run
while something else is training.

What each chart is for, because a training curve is a diagnosis and not a grade:

* **CNN** — the gap between train and validation loss is the whole story. Train
  loss falling while validation loss climbs means the model is memorizing
  variants of documents it has already seen, which is the failure mode this
  dataset is most prone to: 12 variants come from every one source scan.
* **Siamese** — the distance between mean positive and mean negative similarity
  is what ``highest_similarity`` actually thresholds at request time. A curve
  that drives loss down without widening that gap has learned nothing useful.
* **Dataset** — per-op and per-kind counts, because a tamper classifier trained
  on a dataset that is 78% one document kind will report an accuracy that
  belongs to that kind and not to the other.

Charts contain aggregate numbers only — no document imagery, no field values —
so unlike the scans they are safe to put in a report. They are written next to
the metrics they describe, under ``saved_models/``, which is git-ignored.

Usage::

    python -m ml_training.plot_metrics
    python -m ml_training.plot_metrics --which cnn --dpi 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from . import config

# Palette kept consistent across charts: train is muted, validation is loud,
# because validation is the number you are allowed to believe.
TRAIN_COLOR = "#94a3b8"
VAL_COLOR = "#2563eb"
GOOD_COLOR = "#16a34a"
BAD_COLOR = "#dc2626"
BEST_COLOR = "#f59e0b"
GRID_KWARGS = {"alpha": 0.3, "linewidth": 0.6}


def _pyplot():
    """Import pyplot with a non-interactive backend, or explain what is missing.

    ``Agg`` is selected before pyplot is imported because this project is
    headless by policy (it pins ``opencv-python-headless``); without it,
    matplotlib may try to find a display and fail on a server.
    """
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise SystemExit(
            "matplotlib is not installed. It is needed only for this script:\n"
            "  pip install matplotlib==3.9.2"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _series(history: list[dict], key: str) -> list[float]:
    """Pull one metric out of the epoch history, with missing values as NaN.

    ``train_siamese`` writes NaN for ``pair_auc`` when a validation split has no
    negative pairs to compare, so NaN is expected data here rather than an
    error. matplotlib leaves a gap where a NaN is, which is the honest rendering.
    """
    out = []
    for row in history:
        value = row.get(key)
        out.append(float("nan") if value is None else float(value))
    return out


def _all_nan(values: list[float]) -> bool:
    return all(math.isnan(v) for v in values)


def _note_unmeasurable(axis, message: str) -> None:
    """Say why a panel is empty instead of leaving a blank axis."""
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
        wrap=True,
    )


def _mark_best(axis, best_epoch: int | None, label: str = "best") -> None:
    if not best_epoch:
        return
    axis.axvline(
        best_epoch, color=BEST_COLOR, linestyle="--", linewidth=1.2, zorder=0
    )
    axis.annotate(
        f"{label} (epoch {best_epoch})",
        xy=(best_epoch, 0.02),
        xycoords=("data", "axes fraction"),
        rotation=90,
        fontsize=7,
        color=BEST_COLOR,
        ha="right",
        va="bottom",
    )


def _finish(axis, title: str, ylabel: str, xlabel: str = "epoch") -> None:
    axis.set_title(title, fontsize=10, loc="left")
    axis.set_xlabel(xlabel, fontsize=8)
    axis.set_ylabel(ylabel, fontsize=8)
    axis.tick_params(labelsize=8)
    axis.grid(True, **GRID_KWARGS)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=7, framealpha=0.9)


# --- CNN --------------------------------------------------------------------
def plot_cnn(metrics: dict, out_path: Path, dpi: int) -> Path:
    plt = _pyplot()
    history = metrics.get("history") or []
    epochs = [row["epoch"] for row in history]
    best = metrics.get("best_epoch")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # 1. Loss. The train/val gap is the overfitting diagnosis.
    train_loss = _series(history, "train_loss")
    val_loss = _series(history, "loss")
    axes[0].plot(epochs, train_loss, color=TRAIN_COLOR, marker="o",
                 markersize=3, label="train loss")
    axes[0].plot(epochs, val_loss, color=VAL_COLOR, marker="o",
                 markersize=3, label="validation loss")
    _mark_best(axes[0], best)
    _finish(axes[0], "Loss - train vs validation", "BCE loss")

    # A gap this chart exists to make obvious, quantified in the corner.
    if train_loss and val_loss and not _all_nan(train_loss):
        gap = val_loss[-1] - train_loss[-1]
        axes[0].annotate(
            f"final gap {gap:+.3f}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            color=BAD_COLOR if gap > 0.1 else "#64748b",
        )

    # 2. What the detector is judged on. AUC is threshold-free; accuracy is not.
    auc = _series(history, "auc")
    accuracy = _series(history, "accuracy")
    axes[1].plot(epochs, auc, color=GOOD_COLOR, marker="o", markersize=3,
                 label="validation ROC AUC")
    axes[1].plot(epochs, accuracy, color=VAL_COLOR, marker="o", markersize=3,
                 linestyle="--", label="validation accuracy")
    axes[1].axhline(0.5, color=BAD_COLOR, linewidth=1, linestyle=":",
                    label="0.5 = coin flip")
    axes[1].set_ylim(0.0, 1.02)
    _mark_best(axes[1], best)
    _finish(axes[1], "Validation performance", "score")

    final = metrics.get("final_val") or {}
    if final:
        axes[1].annotate(
            f"best AUC {max((v for v in auc if not math.isnan(v)), default=float('nan')):.3f}"
            f"\nn = {final.get('count', '?')} val images",
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#334155",
        )

    # 3. The operating point. This is the number that informs _CNN_CONCERN.
    threshold = _series(history, "best_threshold")
    threshold_accuracy = _series(history, "best_threshold_accuracy")
    axes[2].plot(epochs, threshold, color="#7c3aed", marker="o", markersize=3,
                 label="best P(tampered) cut-off")
    axes[2].plot(epochs, threshold_accuracy, color=GOOD_COLOR, marker="o",
                 markersize=3, linestyle="--", label="accuracy at that cut-off")
    axes[2].axhline(0.5, color="#94a3b8", linewidth=1, linestyle=":",
                    label="0.5 = the default")
    axes[2].set_ylim(0.0, 1.02)
    _mark_best(axes[2], best)
    _finish(axes[2], "Operating point", "probability / accuracy")

    subtitle = (
        f"ResNet-18, pretrained={metrics.get('pretrained')}, "
        f"frozen backbone={metrics.get('freeze_backbone')} | "
        f"train {metrics.get('train_count')} "
        f"(pos {metrics.get('train_positives')}) / val {metrics.get('val_count')} | "
        f"{metrics.get('epochs_run')} epochs run"
    )
    figure.suptitle("CNN tamper classifier", fontsize=12, x=0.01, y=0.975, ha="left")
    figure.text(0.01, 0.915, subtitle, fontsize=8, color="#475569", ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.885))
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


# --- Siamese ----------------------------------------------------------------
def plot_siamese(metrics: dict, out_path: Path, dpi: int) -> Path:
    plt = _pyplot()
    history = metrics.get("history") or []
    epochs = [row["epoch"] for row in history]
    best = metrics.get("best_epoch")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    train_loss = _series(history, "train_loss")
    val_loss = _series(history, "loss")
    axes[0].plot(epochs, train_loss, color=TRAIN_COLOR, marker="o",
                 markersize=3, label="train loss")
    axes[0].plot(epochs, val_loss, color=VAL_COLOR, marker="o",
                 markersize=3, label="validation loss")
    _mark_best(axes[0], best)
    _finish(axes[0], "Loss - train vs validation", "cosine embedding loss")

    # 2. The separation the detector thresholds. Falling loss with a flat gap
    # means the embedder is not actually telling documents apart.
    positive = _series(history, "mean_positive_similarity")
    negative = _series(history, "mean_negative_similarity")
    if _all_nan(negative):
        axes[1].plot(epochs, positive, color=GOOD_COLOR, marker="o",
                     markersize=3, label="same document")
        _note_unmeasurable(
            axes[1],
            "no different-document pairs in the\nvalidation split, so separation\n"
            "cannot be measured - raise --val-pairs\nor add more source scans",
        )
    else:
        axes[1].plot(epochs, positive, color=GOOD_COLOR, marker="o",
                     markersize=3, label="same document")
        axes[1].plot(epochs, negative, color=BAD_COLOR, marker="o",
                     markersize=3, label="different documents")
        axes[1].fill_between(epochs, negative, positive, color=GOOD_COLOR,
                             alpha=0.12, label="separation")
        gap = [p - n for p, n in zip(positive, negative)]
        finite = [g for g in gap if not math.isnan(g)]
        if finite:
            axes[1].annotate(
                f"final separation {gap[-1]:+.3f}",
                xy=(0.98, 0.05),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=7,
                color=GOOD_COLOR if gap[-1] > 0.1 else BAD_COLOR,
            )
    margin = metrics.get("margin")
    if margin is not None:
        axes[1].axhline(float(margin), color="#94a3b8", linewidth=1,
                        linestyle=":", label=f"loss margin {margin}")
    axes[1].set_ylim(-1.02, 1.02)
    _mark_best(axes[1], best)
    _finish(axes[1], "Cosine similarity by pair type", "mean cosine similarity")

    # 3. Pair AUC plus the threshold that separated best - the evidence for
    # _SIAMESE_CONCERN, which was originally set against mock scores.
    pair_auc = _series(history, "pair_auc")
    threshold = _series(history, "best_threshold")
    threshold_accuracy = _series(history, "best_threshold_accuracy")
    if _all_nan(pair_auc):
        _note_unmeasurable(
            axes[2],
            "pair AUC needs both positive and\nnegative pairs in validation",
        )
    else:
        axes[2].plot(epochs, pair_auc, color=GOOD_COLOR, marker="o",
                     markersize=3, label="pair ROC AUC")
        axes[2].axhline(0.5, color=BAD_COLOR, linewidth=1, linestyle=":",
                        label="0.5 = coin flip")
    axes[2].plot(epochs, threshold, color="#7c3aed", marker="o", markersize=3,
                 linestyle="--", label="best similarity cut-off")
    axes[2].plot(epochs, threshold_accuracy, color=VAL_COLOR, marker="o",
                 markersize=3, linestyle="-.", label="accuracy at that cut-off")
    axes[2].set_ylim(0.0, 1.02)
    _mark_best(axes[2], best)
    _finish(axes[2], "Pair separability and operating point", "score")

    subtitle = (
        f"ResNet-18 embedder (fc = Identity), pretrained={metrics.get('pretrained')}, "
        f"margin={metrics.get('margin')} | "
        f"train rows {metrics.get('train_rows')} / val rows {metrics.get('val_rows')} | "
        f"{metrics.get('epochs_run')} epochs run"
    )
    figure.suptitle("Siamese document embedder", fontsize=12, x=0.01, y=0.975, ha="left")
    figure.text(0.01, 0.915, subtitle, fontsize=8, color="#475569", ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.885))
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


# --- Dataset ----------------------------------------------------------------
def plot_dataset(manifest_path: Path, out_path: Path, dpi: int) -> Path:
    plt = _pyplot()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # 1. How each class was made. An op that dominates is an op the classifier
    # will be best at, which is not the same as being good at tampering.
    ops = Counter(row["op"] for row in rows)
    order = ["genuine"] + sorted(op for op in ops if op != "genuine")
    colors = [GOOD_COLOR] + [BAD_COLOR] * (len(order) - 1)
    bars = axes[0].bar(order, [ops[op] for op in order], color=colors, alpha=0.85)
    axes[0].bar_label(bars, fontsize=7, padding=2)
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    _finish(axes[0], "Variants by operation", "images", xlabel="")

    # 2. Class balance within each split. Both must be near 50/50, or the AUC in
    # the CNN chart is measuring the prior and not the model.
    splits = ["train", "val"]
    genuine = [sum(1 for r in rows if r["split"] == s and r["label"] == "0")
               for s in splits]
    tampered = [sum(1 for r in rows if r["split"] == s and r["label"] == "1")
                for s in splits]
    positions = range(len(splits))
    left = axes[1].bar([p - 0.21 for p in positions], genuine, width=0.38,
                       color=GOOD_COLOR, alpha=0.85, label="genuine")
    right = axes[1].bar([p + 0.21 for p in positions], tampered, width=0.38,
                        color=BAD_COLOR, alpha=0.85, label="tampered")
    axes[1].bar_label(left, fontsize=7, padding=2)
    axes[1].bar_label(right, fontsize=7, padding=2)
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(
        [f"{s}\n{sum(1 for r in rows if r['split'] == s)} images" for s in splits]
    )
    _finish(axes[1], "Class balance per split", "images", xlabel="")

    # 3. Document kinds, by variant and by distinct source. The source count is
    # the one that limits what the model can generalize to.
    kinds = sorted({row["doc_kind"] for row in rows})
    variant_counts = [sum(1 for r in rows if r["doc_kind"] == k) for k in kinds]
    source_counts = [len({r["source"] for r in rows if r["doc_kind"] == k})
                     for k in kinds]
    positions = range(len(kinds))
    variants = axes[2].bar([p - 0.2 for p in positions], variant_counts,
                           width=0.4, color=VAL_COLOR, alpha=0.85,
                           label="generated variants")
    sources = axes[2].bar([p + 0.2 for p in positions], source_counts, width=0.4,
                          color=BEST_COLOR, alpha=0.9, label="distinct source scans")
    axes[2].bar_label(variants, fontsize=7, padding=2)
    axes[2].bar_label(sources, fontsize=7, padding=2)
    axes[2].set_xticks(list(positions))
    axes[2].set_xticklabels(kinds)
    axes[2].set_yscale("log")
    _finish(axes[2], "Variants vs real sources, by kind", "images (log scale)",
            xlabel="")

    total_sources = len({row["source"] for row in rows})
    per_source = len(rows) / total_sources if total_sources else 0
    subtitle = (
        f"{len(rows)} images from {total_sources} source scans "
        f"({per_source:.0f} variants per source) | "
        "split grouped by source, so no source appears in both train and val"
    )
    figure.suptitle("Fraud dataset composition", fontsize=12, x=0.01, y=0.975, ha="left")
    figure.text(0.01, 0.915, subtitle, fontsize=8, color="#475569", ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.885))
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


# --- CLI --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--which",
        choices=("all", "cnn", "siamese", "dataset"),
        default="all",
        help="Which chart(s) to draw (default: all that have data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Where to write the PNGs (default: alongside the metrics files).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    config.ensure_dirs()
    out_dir = args.out_dir or config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    missing: list[str] = []

    if args.which in ("all", "cnn"):
        metrics = _load_json(config.CNN_METRICS_PATH)
        if metrics:
            written.append(
                plot_cnn(metrics, out_dir / "cnn_tamper.curves.png", args.dpi)
            )
        else:
            missing.append(
                f"{config.CNN_METRICS_PATH.name} - run: python -m ml_training.train_cnn"
            )

    if args.which in ("all", "siamese"):
        metrics = _load_json(config.SIAMESE_METRICS_PATH)
        if metrics:
            written.append(
                plot_siamese(
                    metrics, out_dir / "siamese_embedder.curves.png", args.dpi
                )
            )
        else:
            missing.append(
                f"{config.SIAMESE_METRICS_PATH.name} - run: "
                "python -m ml_training.train_siamese"
            )

    if args.which in ("all", "dataset"):
        if config.MANIFEST_PATH.is_file():
            written.append(
                plot_dataset(
                    config.MANIFEST_PATH,
                    out_dir / "dataset_composition.png",
                    args.dpi,
                )
            )
        else:
            missing.append(
                f"{config.MANIFEST_PATH.name} - run: "
                "python -m ml_training.prepare_dataset"
            )

    for path in written:
        print(f"  wrote {path.relative_to(config.ROOT_DIR).as_posix()}")
    for note in missing:
        print(f"  skipped {note}")
    if not written:
        raise SystemExit("Nothing to chart yet.")

    print(f"\n{len(written)} chart(s) in {out_dir}")
    print("These hold aggregate numbers only - no scan imagery - so unlike the")
    print("raw documents they are safe to paste into a report.")


if __name__ == "__main__":
    main()
