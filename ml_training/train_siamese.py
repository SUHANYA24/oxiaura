"""Train the Siamese document embedder (BUILD_SPEC Phase 6b).

Produces the weight file that ``app/ai/fraud/siamese_detector.py`` loads via
``FRAUD_SIAMESE_WEIGHTS``.

**What "similar" means here.** ``highest_similarity`` compares an uploaded
document against a bank of known-forgery exemplars and treats a high match as
suspicious. So the embedding has to answer *"is this the same physical document
/ template as one I have seen before?"* — not *"is this tampered?"*. Training
pairs follow from that:

* **positive** — two variants of the **same** source scan (including tampered
  ones: a forged copy of a known template must still match that template).
* **negative** — variants of **different** source scans.

Because positives include tampered variants, the space is deliberately
tamper-*invariant* and identity-*sensitive*. Tamper detection is the CNN's job.

Compatibility with ``siamese_detector.embed``, which builds
``resnet18 -> fc = Identity`` and L2-normalizes the 512-d output:

* No projection head. A head would be trained and then discarded at inference,
  leaving the served embedding space untrained. The loss is applied directly to
  the normalized backbone output instead.
* Saved as a bare ``state_dict`` (loaded with ``strict=False``).

Usage::

    python -m ml_training.train_siamese
    python -m ml_training.train_siamese --epochs 40 --margin 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .train_cnn import load_manifest, roc_auc


def group_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    """Group manifest rows by their source scan."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["source"], []).append(row)
    return groups


def make_pairs(
    rows: list[dict], count: int, rng: random.Random, label: str = "this"
) -> list[tuple[dict, dict, float]]:
    """Build a balanced list of ``(row_a, row_b, target)`` pairs.

    ``target`` is ``+1`` for same-source and ``-1`` for different-source, the
    convention ``nn.CosineEmbeddingLoss`` expects.
    """
    groups = group_by_source(rows)
    # Only sources with two or more variants can produce a positive pair.
    positive_sources = [key for key, items in groups.items() if len(items) >= 2]
    source_keys = list(groups)

    if not positive_sources:
        raise SystemExit(
            "Cannot build positive pairs: every source has fewer than two variants.\n"
            "Increase --aug-per-image in prepare_dataset."
        )
    if len(source_keys) < 2:
        print(
            f"WARNING: the {label} split has only one source scan, so no negative "
            "pairs are possible and every reported similarity is between variants of "
            "the same document. Add more authentic scans."
        )

    pairs: list[tuple[dict, dict, float]] = []
    for index in range(count):
        want_positive = index % 2 == 0 or len(source_keys) < 2
        if want_positive:
            items = groups[rng.choice(positive_sources)]
            first, second = rng.sample(items, 2)
            pairs.append((first, second, 1.0))
        else:
            key_a, key_b = rng.sample(source_keys, 2)
            pairs.append((rng.choice(groups[key_a]), rng.choice(groups[key_b]), -1.0))
    return pairs


class PairDataset(Dataset):
    """Yields ``(image_a, image_b, target)`` for a precomputed pair list."""

    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, row: dict) -> torch.Tensor:
        with Image.open(config.ROOT_DIR / row["path"]) as opened:
            return self.transform(opened.convert("RGB"))

    def __getitem__(self, index: int):
        row_a, row_b, target = self.pairs[index]
        return self._load(row_a), self._load(row_b), torch.tensor(target)


class DocumentEmbedder(nn.Module):
    """ResNet-18 backbone emitting an L2-normalized 512-d embedding.

    ``state_dict()`` keys match a plain ``resnet18`` because the head is
    ``Identity`` — exactly what ``siamese_detector._load_embedder`` rebuilds.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception as exc:  # noqa: BLE001 - first run downloads ~45 MB
            raise SystemExit(
                f"Could not obtain pretrained ResNet-18 weights ({exc}).\n"
                "Connect to the internet for the one-time download, or pass "
                "--no-pretrained."
            )
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(images), dim=1)

    def backbone_state_dict(self):
        """The state dict as the detector expects it (no ``backbone.`` prefix)."""
        return self.backbone.state_dict()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> dict:
    """Report loss plus how well similarity separates same- from different-source."""
    model.eval()
    total_loss, positives, negatives = 0.0, [], []
    seen = 0
    for images_a, images_b, targets in loader:
        images_a, images_b = images_a.to(device), images_b.to(device)
        targets = targets.to(device)
        embed_a, embed_b = model(images_a), model(images_b)
        total_loss += criterion(embed_a, embed_b, targets).item() * images_a.size(0)
        seen += images_a.size(0)

        similarity = torch.sum(embed_a * embed_b, dim=1).cpu().tolist()
        for value, target in zip(similarity, targets.cpu().tolist()):
            (positives if target > 0 else negatives).append(value)

    labels = [1.0] * len(positives) + [0.0] * len(negatives)
    scores = positives + negatives

    # Best split point between the two similarity distributions — a sensible
    # starting value for _SIAMESE_CONCERN in app/services/fraud_service.py.
    threshold, accuracy = 0.5, 0.0
    if scores:
        for candidate in sorted({round(value, 3) for value in scores}):
            correct = sum(
                1 for label, score in zip(labels, scores) if (score >= candidate) == (label > 0)
            )
            if correct / len(scores) > accuracy:
                threshold, accuracy = candidate, correct / len(scores)

    return {
        "loss": total_loss / max(1, seen),
        "mean_positive_similarity": sum(positives) / len(positives) if positives else float("nan"),
        "mean_negative_similarity": sum(negatives) / len(negatives) if negatives else float("nan"),
        "pair_auc": roc_auc(labels, scores),
        "best_threshold": threshold,
        "best_threshold_accuracy": accuracy,
        "pairs": len(scores),
    }


def train(args) -> dict:
    config.set_seed(args.seed)
    config.ensure_dirs()
    device = config.resolve_device(args.device)

    train_rows = load_manifest("train")
    val_rows = load_manifest("val")
    if not train_rows:
        raise SystemExit("Manifest has no training rows - re-run prepare_dataset.")

    model = DocumentEmbedder(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CosineEmbeddingLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Validation pairs are fixed for the whole run so the metric is comparable
    # epoch to epoch; training pairs are resampled each epoch for variety.
    val_loader = None
    if val_rows:
        try:
            val_pairs = make_pairs(
                val_rows, args.val_pairs, random.Random(args.seed + 999), label="validation"
            )
            val_loader = DataLoader(
                PairDataset(val_pairs, config.build_eval_transform()),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
        except SystemExit as exc:
            print(f"WARNING: skipping validation - {exc}")

    print(
        f"device={device}  train_rows={len(train_rows)}  val_rows={len(val_rows)}  "
        f"sources={len(group_by_source(train_rows))}"
    )

    best_metric, best_state, best_epoch, stale = -math.inf, None, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        pairs = make_pairs(
            train_rows, args.pairs_per_epoch, random.Random(args.seed + epoch), label="training"
        )
        loader = DataLoader(
            PairDataset(pairs, config.build_train_transform()),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

        model.train()
        running_loss, seen = 0.0, 0
        for images_a, images_b, targets in loader:
            images_a, images_b = images_a.to(device), images_b.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images_a), model(images_b), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images_a.size(0)
            seen += images_a.size(0)

        train_loss = running_loss / max(1, seen)
        metrics = (
            evaluate(model, val_loader, criterion, device)
            if val_loader
            else {"loss": float("nan"), "pair_auc": float("nan")}
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        auc = metrics["pair_auc"]
        print(
            f"epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  "
            f"pair_auc={'n/a' if math.isnan(auc) else format(auc, '.3f')}"
        )

        score = -train_loss if val_loader is None else (
            auc if not math.isnan(auc) else -metrics["loss"]
        )
        if score > best_metric:
            best_metric, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.backbone_state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs")
                break

    if best_state is None:  # pragma: no cover - only if epochs == 0
        raise SystemExit("Training produced no weights.")

    torch.save(best_state, config.SIAMESE_WEIGHTS_PATH)

    # Evaluate the weights that were just written, not whatever the last epoch
    # left in memory. Without this restore, ``final_val`` describes epoch
    # ``epochs_run`` while ``best_epoch`` names a different one, so the summary
    # line reports a quality the saved file does not have — and early stopping
    # guarantees the two differ by at least ``patience`` epochs.
    model.backbone.load_state_dict(best_state)
    model.to(device)
    final = (
        evaluate(model, val_loader, criterion, device)
        if val_loader
        else {"note": "no validation pairs"}
    )
    summary = {
        "best_epoch": best_epoch,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "pretrained": not args.no_pretrained,
        "margin": args.margin,
        "epochs_run": len(history),
        "final_val": final,
        "history": history,
    }
    config.SIAMESE_METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=config.SIAMESE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.SIAMESE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.SIAMESE_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.SIAMESE_WEIGHT_DECAY)
    parser.add_argument("--margin", type=float, default=config.SIAMESE_MARGIN)
    parser.add_argument("--pairs-per-epoch", type=int, default=config.SIAMESE_PAIRS_PER_EPOCH)
    parser.add_argument("--val-pairs", type=int, default=128)
    parser.add_argument("--patience", type=int, default=config.SIAMESE_PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    summary = train(args)
    final = summary["final_val"]

    print(f"\nSaved weights: {config.SIAMESE_WEIGHTS_PATH}")
    print(f"Saved metrics: {config.SIAMESE_METRICS_PATH}")
    if isinstance(final, dict) and "pair_auc" in final:
        print(
            f"Best epoch {summary['best_epoch']}  "
            f"mean same-doc similarity={final['mean_positive_similarity']:.3f}  "
            f"mean different-doc={final['mean_negative_similarity']:.3f}  "
            f"suggested similarity cut-off={final['best_threshold']:.3f}"
        )
    print(
        "\nTo use it, set this in .env and restart the app/worker:\n"
        f"  FRAUD_SIAMESE_WEIGHTS="
        f"{Path(config.SIAMESE_WEIGHTS_PATH).relative_to(config.ROOT_DIR).as_posix()}\n"
        "Then build the reference bank: python -m ml_training.build_reference_bank"
    )


if __name__ == "__main__":
    main()
