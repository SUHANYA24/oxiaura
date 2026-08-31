"""Fraud detection subpackage (BUILD_SPEC Phase 6).

Three independent detectors — ELA (real), CNN classifier, and Siamese
similarity — each returning a normalized score. The CNN and Siamese detectors
fall back to deterministic mock scores when no trained weights are configured,
so the whole pipeline and API are testable without them.

State after Phase 6b, measured by ``ml_training.evaluate_pipeline`` on 516
held-out images:

* **ELA** — real, but AUC 0.485 on these documents, i.e. no usable signal.
* **CNN** — real, AUC 0.830. The only detector currently separating the classes.
* **Siamese** — trained, but still on its mock (AUC 0.474) because it also needs
  a reference bank of confirmed forgeries and there is none.

A mock score is stable per file but unrelated to tampering, so it dilutes the
aggregate rather than abstaining from it. See ``ml_training/README.md`` §8.
"""
