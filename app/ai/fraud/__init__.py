"""Fraud detection subpackage (BUILD_SPEC Phase 6).

Three independent detectors — ELA (real), CNN classifier, and Siamese
similarity — each returning a normalized score. The CNN and Siamese detectors
fall back to deterministic mock scores when no trained weights are configured,
so the whole pipeline and API are testable before Phase 6b swaps in real models.
"""
