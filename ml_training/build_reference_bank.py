"""Build the Siamese reference embedding bank (BUILD_SPEC Phase 6b).

``siamese_detector.highest_similarity`` scores an upload by its closest match
against a bank of *known-forgery exemplars*: confirmed fakes, recycled
templates, documents you have already rejected. This script turns a folder of
such exemplars into that bank.

It deliberately calls :func:`app.ai.fraud.siamese_detector.embed` rather than
re-implementing the transform, so the bank can never drift out of step with the
embedding the detector computes at request time.

Output mirrors the trainers' convention — a bare tensor the consumer can iterate
directly, with provenance in a sibling JSON:

* ``saved_models/reference_embeddings.pt``   float tensor ``[N, 512]``
* ``saved_models/reference_embeddings.json`` source filenames + settings

.. warning::
   Building the bank is not enough to make the detector use it.
   ``fraud_service.run_detectors`` currently calls
   ``siamese_detector.highest_similarity(file_path)`` with no
   ``known_embeddings`` argument, so the detector always takes its mock path.
   Loading this file and passing it through is a separate change — see
   ml_training/README.md, "Wiring the Siamese bank into the app".

Usage::

    python -m ml_training.build_reference_bank
    python -m ml_training.build_reference_bank --source-dir path/to/known/fakes
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.ai.fraud import siamese_detector

from . import config


def collect_exemplars(source_dir: Path) -> list[Path]:
    """Return the exemplar image paths under ``source_dir`` (recursively)."""
    if not source_dir.is_dir():
        raise SystemExit(
            f"No exemplar folder at {source_dir}.\n"
            "Create it and add images of confirmed forgeries / recycled templates, "
            "one file each."
        )
    paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(
            f"{source_dir} contains no images.\n"
            "Add at least one confirmed forgery before building the bank."
        )
    return paths


def build(source_dir: Path, weights_path: Path) -> dict:
    """Embed every exemplar and write the bank plus its metadata."""
    if not weights_path.is_file():
        raise SystemExit(
            f"No trained embedder at {weights_path}.\n"
            "Run: python -m ml_training.train_siamese"
        )

    config.ensure_dirs()
    paths = collect_exemplars(source_dir)

    vectors, sources = [], []
    for path in paths:
        # embed() takes the weights path directly and never touches current_app,
        # so no Flask application context is needed here.
        vectors.append(siamese_detector.embed(str(path), str(weights_path)))
        sources.append(path.relative_to(source_dir).as_posix())
        print(f"  embedded {sources[-1]}")

    bank = torch.stack(vectors)
    torch.save(bank, config.REFERENCE_BANK_PATH)

    # Near-identical exemplars add cost without adding coverage; surface them.
    redundant = []
    if len(vectors) > 1:
        similarity = bank @ bank.T
        similarity.fill_diagonal_(-1.0)
        seen: set[frozenset[str]] = set()
        for index, name in enumerate(sources):
            closest = int(torch.argmax(similarity[index]))
            score = float(similarity[index, closest])
            pair = frozenset((name, sources[closest]))
            # A mutual nearest pair would otherwise be reported twice.
            if score > 0.98 and pair not in seen:
                seen.add(pair)
                redundant.append({"a": name, "b": sources[closest], "similarity": round(score, 4)})

    metadata = {
        "count": len(sources),
        "embedding_dim": int(bank.shape[1]),
        "sources": sources,
        "source_dir": source_dir.as_posix(),
        "weights": weights_path.as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "near_duplicates": redundant,
    }
    config.REFERENCE_BANK_PATH.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=config.REFERENCE_DIR,
        help="Folder of confirmed-forgery exemplars.",
    )
    parser.add_argument("--weights", type=Path, default=config.SIAMESE_WEIGHTS_PATH)
    args = parser.parse_args()

    metadata = build(args.source_dir, args.weights)

    print(f"\nSaved bank:     {config.REFERENCE_BANK_PATH}  ({metadata['count']} exemplars)")
    print(f"Saved metadata: {config.REFERENCE_BANK_PATH.with_suffix('.json')}")
    for pair in metadata["near_duplicates"]:
        print(
            f"  note: '{pair['a']}' and '{pair['b']}' are {pair['similarity']:.3f} similar "
            "- one of them is probably redundant"
        )
    print(
        "\nReminder: the app does not consult this bank yet. See "
        "ml_training/README.md -> 'Wiring the Siamese bank into the app'."
    )


if __name__ == "__main__":
    main()
