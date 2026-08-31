"""CNN tamper classifier (BUILD_SPEC Phase 6).

Architecture is a ResNet-18 with a binary head returning P(tampered) in [0, 1].
Trained weights are loaded from ``FRAUD_CNN_WEIGHTS`` when configured; until
Phase 6b provides them, ``predict_tamper_probability`` returns a deterministic
mock derived from the file's SHA-256 so the pipeline and API are fully testable.
The heavy torch import happens only on the real-inference path.
"""

from __future__ import annotations

import hashlib
import threading

from flask import current_app

_MOCK_SALT = b"cnn-tamper-v1"
_model = None
_model_lock = threading.Lock()


def _deterministic_mock(file_path: str) -> float:
    """A stable pseudo-probability in [0, 1] keyed on the file contents."""
    with open(file_path, "rb") as handle:
        digest = hashlib.sha256(_MOCK_SALT + handle.read()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _load_model(weights_path: str):
    """Build ResNet-18 with a 1-logit head and load trained weights."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from torchvision import models

                net = models.resnet18(weights=None)
                net.fc = torch.nn.Linear(net.fc.in_features, 1)
                state = torch.load(weights_path, map_location="cpu")
                net.load_state_dict(state)
                net.eval()
                _model = net
    return _model


def _run_inference(file_path: str, weights_path: str) -> float:
    import torch
    from PIL import Image
    from torchvision import transforms

    model = _load_model(weights_path)
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
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
    return float(prob)


def predict_tamper_probability(file_path: str) -> float:
    """Return P(document is tampered) in [0, 1].

    Uses the trained model when ``FRAUD_CNN_WEIGHTS`` points at a readable file;
    otherwise returns a deterministic mock score.
    """
    import os

    weights_path = current_app.config.get("FRAUD_CNN_WEIGHTS")
    if weights_path and os.path.isfile(weights_path):
        return round(_run_inference(file_path, weights_path), 4)
    return round(_deterministic_mock(file_path), 4)
