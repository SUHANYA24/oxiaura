"""Unit tests for the three fraud detectors (ELA real, CNN/Siamese mock)."""

from io import BytesIO

import pytest
from PIL import Image

from app.ai.fraud import cnn_classifier, ela_detector, siamese_detector


def _write_png(path, color="white", size=(120, 120)):
    img = Image.new("RGB", size, color)
    img.save(path, format="PNG")
    return str(path)


class TestEla:
    def test_score_in_range_with_details(self, tmp_path):
        path = _write_png(tmp_path / "clean.png")
        score, details = ela_detector.score_image(path)
        assert 0.0 <= score <= 100.0
        assert "ela_component" in details
        assert details["exif_penalty"] == 0.0

    def test_deterministic(self, tmp_path):
        path = _write_png(tmp_path / "d.png")
        assert ela_detector.score_image(path)[0] == ela_detector.score_image(path)[0]


class TestCnnMock:
    def test_probability_in_unit_range(self, app, tmp_path):
        path = _write_png(tmp_path / "c.png")
        prob = cnn_classifier.predict_tamper_probability(path)
        assert 0.0 <= prob <= 1.0

    def test_deterministic_per_file(self, app, tmp_path):
        p1 = _write_png(tmp_path / "a.png", color="white")
        p2 = _write_png(tmp_path / "b.png", color="black")
        assert cnn_classifier.predict_tamper_probability(p1) == (
            cnn_classifier.predict_tamper_probability(p1)
        )
        # Different contents should generally yield different mock scores.
        assert cnn_classifier.predict_tamper_probability(p1) != (
            cnn_classifier.predict_tamper_probability(p2)
        )


class TestSiameseMock:
    def test_similarity_in_unit_range(self, app, tmp_path):
        path = _write_png(tmp_path / "s.png")
        sim = siamese_detector.highest_similarity(path)
        assert 0.0 <= sim <= 1.0

    def test_deterministic_per_file(self, app, tmp_path):
        path = _write_png(tmp_path / "s.png")
        assert siamese_detector.highest_similarity(path) == (
            siamese_detector.highest_similarity(path)
        )
