"""Siamese similarity detector (BUILD_SPEC Phase 6).

Embeds a document image and returns the highest cosine similarity against a set
of known reference embeddings (e.g. templates of previously confirmed forgeries).
A high similarity to a known-fraud template is itself suspicious.

Trained embedding weights come from ``FRAUD_SIAMESE_WEIGHTS``; until Phase 6b
provides them (and a populated reference set), ``highest_similarity`` returns a
deterministic mock keyed on the file contents.
"""

from __future__ import annotations

import hashlib
import threading

from flask import current_app

_MOCK_SALT = b"siamese-sim-v1"
_embedder = None
_embedder_lock = threading.Lock()


def _deterministic_mock(file_path: str) -> float:
    """A stable pseudo-similarity in [0, 1] keyed on the file contents."""
    with open(file_path, "rb") as handle:
        digest = hashlib.sha256(_MOCK_SALT + handle.read()).hexdigest()
    return int(digest[8:16], 16) / 0xFFFFFFFF


def _load_embedder(weights_path: str):
    """ResNet-18 backbone with the classifier head removed → embedding vector."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                import torch
                from torchvision import models

                net = models.resnet18(weights=None)
                net.fc = torch.nn.Identity()
                state = torch.load(weights_path, map_location="cpu")
                net.load_state_dict(state, strict=False)
                net.eval()
                _embedder = net
    return _embedder


def embed(file_path: str, weights_path: str):
    """Return the L2-normalized embedding vector for an image (real model)."""
    import torch
    from PIL import Image
    from torchvision import transforms

    model = _load_embedder(weights_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    with Image.open(file_path) as opened:
        tensor = preprocess(opened.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vector = model(tensor).squeeze(0)
    return torch.nn.functional.normalize(vector, dim=0)


def highest_similarity(file_path: str, known_embeddings=None) -> float:
    """Return the highest cosine similarity in [0, 1] against known embeddings.

    Falls back to a deterministic mock when no trained weights are configured or
    no reference embeddings are available (the Phase 6 default).
    """
    import os

    weights_path = current_app.config.get("FRAUD_SIAMESE_WEIGHTS")
    if weights_path and os.path.isfile(weights_path) and known_embeddings:
        import torch

        vector = embed(file_path, weights_path)
        sims = [
            float(torch.dot(vector, ref).clamp(-1.0, 1.0)) for ref in known_embeddings
        ]
        best = max(sims) if sims else 0.0
        return round(max(0.0, best), 4)
    return round(_deterministic_mock(file_path), 4)
