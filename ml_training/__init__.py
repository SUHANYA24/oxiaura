"""Offline model training package (BUILD_SPEC Phase 6b).

Nothing in here is imported by the Flask app or the request cycle — these are
command-line scripts you run by hand to build datasets and produce the weight
files that ``app/ai/fraud`` loads at inference time.

Run every script as a module from the repository root, e.g.::

    python -m ml_training.prepare_dataset
    python -m ml_training.train_cnn
"""
