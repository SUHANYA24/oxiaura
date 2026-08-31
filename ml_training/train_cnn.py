"""Train the CNN tamper classifier (BUILD_SPEC Phase 6b).

Produces the weight file that ``app/ai/fraud/cnn_classifier.py`` loads via the
``FRAUD_CNN_WEIGHTS`` setting, replacing its deterministic mock score.

Three details are load-bearing for compatibility with the detector, which does
``resnet18 -> fc = Linear(in_features, 1) -> load_state_dict(state) ->
sigmoid(logit)``:

1. The architecture must be ResNet-18 with a **single-logit** ``fc`` head.
2. The file must be a **bare ``state_dict``**, not a ``{"model": ..., "epoch": ...}``
   checkpoint — the detector calls ``load_state_dict`` on whatever it loads, with
   ``strict=True``. Metrics go to a sibling ``.metrics.json`` instead.
3. Input normalization must match :data:`ml_training.config.IMAGENET_MEAN` /
   ``IMAGENET_STD``, which mirror the detector's own transform.

The model is trained with ``BCEWithLogitsLoss`` on the raw logit because the
detector applies the sigmoid itself.

Usage::

    python -m ml_training.train_cnn
    python -m ml_training.train_cnn --epochs 30 --freeze-backbone
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config


def load_manifest(split: str | None = None) -> list[dict]:
    """Read ``datasets/fraud/manifest.csv``, optionally filtered by split."""
    if not config.MANIFEST_PATH.is_file():
        raise SystemExit(
            f"No manifest at {config.MANIFEST_PATH}.\n"
            "Run: python -m ml_training.prepare_dataset"
        )
    with config.MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["label"] = int(row["label"])
    if split:
        rows = [row for row in rows if row["split"] == split]
    return rows


class TamperDataset(Dataset):
    """Manifest-backed image/label dataset.

    Images are opened with PIL and converted to RGB, matching the detector's
    inference path so training and serving see identical pixel data.
    """

    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = config.ROOT_DIR / row["path"]
        with Image.open(path) as opened:
            tensor = self.transform(opened.convert("RGB"))
        return tensor, torch.tensor([float(row["label"])])


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-18 with a 1-logit head — the exact shape the detector rebuilds."""
    from torchvision import models

    weights = "IMAGENET1K_V1" if pretrained else None
    try:
        net = models.resnet18(weights=weights)
    except Exception as exc:  # noqa: BLE001 - first run downloads ~45 MB
        raise SystemExit(
            f"Could not obtain pretrained ResNet-18 weights ({exc}).\n"
            "Either connect to the internet for the one-time download, or pass "
            "--no-pretrained (expect noticeably worse results on a small dataset)."
        )

    if freeze_backbone:
        for parameter in net.parameters():
            parameter.requires_grad = False

    # Assigned after freezing so the new head always stays trainable.
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def roc_auc(labels: list[float], scores: list[float]) -> float:
    """Rank-based ROC AUC (avoids a scikit-learn dependency).

    Returns NaN when only one class is present, which is common on a tiny
    validation split.
    """
    positives = [s for label, s in zip(labels, scores) if label >= 0.5]
    negatives = [s for label, s in zip(labels, scores) if label < 0.5]
    if not positives or not negatives:
        return float("nan")

    ranked = sorted(zip(scores, labels))
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ranked):
        stop = index
        while stop + 1 < len(ranked) and ranked[stop + 1][0] == ranked[index][0]:
            stop += 1
        # Ties share the average rank.
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1

    positive_rank_sum = sum(
        ranks[position] for position, (_, label) in enumerate(ranked) if label >= 0.5
    )
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_threshold(labels: list[float], scores: list[float]) -> tuple[float, float]:
    """Sweep candidate cut-offs and return ``(threshold, accuracy)``.

    Reported so you can set a sensible per-signal concern level in
    ``app/services/fraud_service.py`` (``_CNN_CONCERN``) from evidence rather
    than the 0.5 default.
    """
    if not scores:
        return 0.5, 0.0
    candidates = sorted({round(s, 3) for s in scores} | {0.5})
    best = (0.5, -1.0)
    for threshold in candidates:
        correct = sum(
            1
            for label, score in zip(labels, scores)
            if (score >= threshold) == (label >= 0.5)
        )
        accuracy = correct / len(labels)
        if accuracy > best[1]:
            best = (threshold, accuracy)
    return best


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> dict:
    """Return loss, accuracy, AUC, and the raw scores for one split."""
    model.eval()
    total_loss, labels, scores = 0.0, [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * images.size(0)
        scores.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
        labels.extend(targets.squeeze(1).cpu().tolist())

    count = max(1, len(labels))
    threshold, accuracy = best_threshold(labels, scores)
    default_accuracy = sum(
        1 for label, score in zip(labels, scores) if (score >= 0.5) == (label >= 0.5)
    ) / count
    return {
        "loss": total_loss / count,
        "accuracy": default_accuracy,
        "auc": roc_auc(labels, scores),
        "best_threshold": threshold,
        "best_threshold_accuracy": accuracy,
        "count": len(labels),
    }


def train(args) -> dict:
    config.set_seed(args.seed)
    config.ensure_dirs()
    device = config.resolve_device(args.device)

    train_rows = load_manifest("train")
    val_rows = load_manifest("val")
    if not train_rows:
        raise SystemExit("Manifest has no training rows - re-run prepare_dataset.")
    if not val_rows:
        print(
            "WARNING: no validation rows (too few source images). Training will run "
            "but the reported metrics are meaningless."
        )

    train_loader = DataLoader(
        TamperDataset(train_rows, config.build_train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        TamperDataset(val_rows, config.build_eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(
        pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone
    ).to(device)

    # Counter any class imbalance rather than letting the model favour the
    # majority class.
    positives = sum(row["label"] for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight = torch.tensor(
        [negatives / positives if positives else 1.0], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        f"device={device}  train={len(train_rows)} (pos={positives}, neg={negatives})  "
        f"val={len(val_rows)}"
    )

    best_metric, best_state, best_epoch, stale = -math.inf, None, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(1, len(train_rows))
        metrics = (
            evaluate(model, val_loader, criterion, device)
            if val_rows
            else {"loss": float("nan"), "accuracy": float("nan"), "auc": float("nan")}
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        auc_text = "n/a" if math.isnan(metrics["auc"]) else f"{metrics['auc']:.3f}"
        print(
            f"epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  val_acc={metrics['accuracy']:.3f}  "
            f"val_auc={auc_text}"
        )

        # Prefer AUC; fall back to negative loss when AUC is undefined.
        score = metrics["auc"] if not math.isnan(metrics["auc"]) else -metrics["loss"]
        if not val_rows:
            score = -train_loss
        if score > best_metric:
            best_metric, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs")
                break

    if best_state is None:  # pragma: no cover - only if epochs == 0
        raise SystemExit("Training produced no weights.")

    # Bare state_dict — see the module docstring, point 2.
    torch.save(best_state, config.CNN_WEIGHTS_PATH)

    model.load_state_dict(best_state)
    model.to(device)
    final = (
        evaluate(model, val_loader, criterion, device)
        if val_rows
        else {"note": "no validation split"}
    )
    summary = {
        "best_epoch": best_epoch,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_positives": positives,
        "pretrained": not args.no_pretrained,
        "freeze_backbone": args.freeze_backbone,
        "epochs_run": len(history),
        "final_val": final,
        "history": history,
    }
    config.CNN_METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CNN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.CNN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.CNN_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=config.CNN_PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)."
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the new head - a reasonable choice on a very small dataset.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Start from random weights instead of ImageNet.",
    )
    args = parser.parse_args()

    summary = train(args)
    final = summary["final_val"]

    print(f"\nSaved weights: {config.CNN_WEIGHTS_PATH}")
    print(f"Saved metrics: {config.CNN_METRICS_PATH}")
    if isinstance(final, dict) and "accuracy" in final:
        auc = final["auc"]
        print(
            f"Best epoch {summary['best_epoch']}  val_acc={final['accuracy']:.3f}  "
            f"val_auc={'n/a' if math.isnan(auc) else format(auc, '.3f')}  "
            f"suggested P(tampered) cut-off={final['best_threshold']:.2f}"
        )
    print(
        "\nTo use it, set this in .env and restart the app/worker:\n"
        f"  FRAUD_CNN_WEIGHTS={Path(config.CNN_WEIGHTS_PATH).relative_to(config.ROOT_DIR).as_posix()}"
    )


if __name__ == "__main__":
    main()
