"""Unit tests for template-based OCR extraction (synthetic images, no EasyOCR)."""

import json

import cv2
import numpy as np
import pytest

from app.ai.ocr import template_extractor as te


def _write_image(path, image):
    """Write an image the same way the module reads it (imencode + bytes)."""
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return path


def _textured(width=320, height=240, seed=7):
    """A feature-rich image: random 16px blocks give ORB plenty of corners."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, size=(height // 16, width // 16, 3), dtype=np.uint8)
    return cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)


def _template(fields=None):
    return {
        "name": "test",
        "fields": fields or {"top": {"box": [0.0, 0.0, 1.0, 0.5]}},
    }


def _stub_ocr(mapping=None, default=None):
    """An ocr_fn returning canned lines, recording every crop it was handed."""
    calls = []

    def ocr_fn(image):
        calls.append(image)
        if mapping is not None:
            return mapping[len(calls) - 1]
        return default if default is not None else [("value", 0.9)]

    ocr_fn.calls = calls
    return ocr_fn


class TestLoadTemplate:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "form.json"
        te.save_template(_template(), path)
        loaded = te.load_template(path)
        assert loaded["fields"]["top"]["box"] == [0.0, 0.0, 1.0, 0.5]

    def test_name_defaults_to_filename(self, tmp_path):
        path = tmp_path / "proposal.json"
        path.write_text(json.dumps({"fields": {"a": {"box": [0, 0, 1, 1]}}}))
        assert te.load_template(path)["name"] == "proposal"

    def test_missing_file(self, tmp_path):
        with pytest.raises(te.TemplateError, match="not found"):
            te.load_template(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(te.TemplateError, match="not valid JSON"):
            te.load_template(path)

    def test_no_fields(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"fields": {}}))
        with pytest.raises(te.TemplateError, match="no 'fields'"):
            te.load_template(path)

    def test_malformed_box(self, tmp_path):
        path = tmp_path / "short.json"
        path.write_text(json.dumps({"fields": {"a": {"box": [0, 0, 1]}}}))
        with pytest.raises(te.TemplateError, match="needs a 'box'"):
            te.load_template(path)

    def test_non_numeric_box(self, tmp_path):
        path = tmp_path / "text.json"
        path.write_text(json.dumps({"fields": {"a": {"box": [0, 0, "1", 1]}}}))
        with pytest.raises(te.TemplateError, match="non-numeric"):
            te.load_template(path)

    def test_accepts_a_quarter_turn_rotation(self, tmp_path):
        path = tmp_path / "sideways.json"
        path.write_text(json.dumps({"rotate": 90, "fields": {"a": {"box": [0, 0, 1, 1]}}}))
        assert te.load_template(path)["rotate"] == 90

    def test_rejects_an_arbitrary_rotation(self, tmp_path):
        """Only lossless quarter turns — an arbitrary angle would resample."""
        path = tmp_path / "tilted.json"
        path.write_text(json.dumps({"rotate": 45, "fields": {"a": {"box": [0, 0, 1, 1]}}}))
        with pytest.raises(te.TemplateError, match="rotate=45"):
            te.load_template(path)


class TestRotateImage:
    def test_zero_is_a_no_op(self):
        image = _textured(64, 48)
        assert np.array_equal(te.rotate_image(image, 0), image)

    def test_quarter_turns_swap_the_axes(self):
        image = _textured(64, 48)
        assert te.rotate_image(image, 90).shape[:2] == (64, 48)
        assert te.rotate_image(image, 270).shape[:2] == (64, 48)
        assert te.rotate_image(image, 180).shape[:2] == (48, 64)

    def test_counter_clockwise_moves_top_right_to_top_left(self):
        """Pins the direction: 90 must undo a card photographed 90 clockwise."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[0, 9] = 255  # top-right corner
        rotated = te.rotate_image(image, 90)
        assert tuple(rotated[0, 0]) == (255, 255, 255)

    def test_four_quarter_turns_return_the_original(self):
        image = _textured(64, 48)
        turned = image
        for _ in range(4):
            turned = te.rotate_image(turned, 90)
        assert np.array_equal(turned, image)

    def test_rejects_an_arbitrary_angle(self):
        with pytest.raises(te.TemplateError, match="counter-clockwise degrees"):
            te.rotate_image(_textured(64, 48), 45)


class TestReadImage:
    def test_reads_bgr(self, tmp_path):
        original = _textured(64, 48)
        image = te.read_image(_write_image(tmp_path / "scan.png", original))
        assert image.shape == (48, 64, 3)

    def test_reads_non_ascii_path(self, tmp_path):
        """cv2.imread returns None for these; the module must not."""
        path = _write_image(tmp_path / "සුපුරුදු-scan.png", _textured(64, 48))
        assert te.read_image(path).shape == (48, 64, 3)

    def test_rejects_non_image(self, tmp_path):
        path = tmp_path / "notes.png"
        path.write_bytes(b"this is not an image")
        with pytest.raises(ValueError, match="Could not decode"):
            te.read_image(path)


class TestCropField:
    def test_crops_the_requested_region(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        image[100:200, :] = 255  # bottom half white
        crop = te.crop_field(image, [0.0, 0.5, 1.0, 0.5], pad=0.0)
        assert crop.shape[:2] == (100, 100)
        assert crop.mean() == 255

    def test_padding_expands_the_box(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        tight = te.crop_field(image, [0.25, 0.25, 0.5, 0.5], pad=0.0)
        padded = te.crop_field(image, [0.25, 0.25, 0.5, 0.5], pad=0.05)
        assert padded.shape[0] > tight.shape[0]

    def test_clamps_to_image_bounds(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = te.crop_field(image, [0.9, 0.9, 0.5, 0.5], pad=0.05)
        assert crop.shape[0] <= 100 and crop.shape[1] <= 100
        assert crop.size > 0

    def test_degenerate_box_returns_placeholder(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = te.crop_field(image, [0.5, 0.5, 0.0, 0.0], pad=0.0)
        assert crop.shape[:2] == (1, 1)


class TestPrepareCrop:
    def test_returns_upscaled_grayscale(self):
        crop = _textured(120, 40)
        prepared = te.prepare_crop(crop)
        assert prepared.ndim == 2
        assert max(prepared.shape) >= te._CROP_MIN_LONG_EDGE


class TestAlignToTemplate:
    def test_aligns_a_shifted_scan(self):
        reference = _textured(320, 240)
        shift = cv2.getRotationMatrix2D((160, 120), 2.0, 1.0)
        shift[:, 2] += (12, 8)
        skewed = cv2.warpAffine(
            reference, shift, (320, 240), borderMode=cv2.BORDER_REPLICATE
        )

        aligned, info = te.align_to_template(skewed, reference)

        assert info["aligned"] is True
        assert info["inliers"] >= te._MIN_MATCHES
        # Alignment must actually reduce the difference from the reference.
        before = float(np.abs(skewed.astype(int) - reference.astype(int)).mean())
        after = float(np.abs(aligned.astype(int) - reference.astype(int)).mean())
        assert after < before

    def test_falls_back_to_resize_when_featureless(self):
        """A blank page has no keypoints — resize and say so, do not warp noise."""
        reference = np.full((240, 320, 3), 255, dtype=np.uint8)
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)

        aligned, info = te.align_to_template(blank, reference)

        assert info["aligned"] is False
        assert info["reason"]
        assert aligned.shape[:2] == reference.shape[:2]

    def test_unrelated_image_is_not_claimed_as_aligned(self):
        reference = _textured(320, 240, seed=1)
        other = _textured(320, 240, seed=99)
        _, info = te.align_to_template(other, reference, min_matches=200)
        assert info["aligned"] is False


class TestExtractFields:
    def test_one_ocr_call_per_field(self):
        image = _textured(200, 200)
        template = _template(
            {
                "a": {"box": [0.0, 0.0, 1.0, 0.3]},
                "b": {"box": [0.0, 0.3, 1.0, 0.3]},
                "c": {"box": [0.0, 0.6, 1.0, 0.3]},
            }
        )
        ocr_fn = _stub_ocr()

        result = te.extract_fields(image, template, ocr_fn=ocr_fn, align=False)

        assert len(ocr_fn.calls) == 3
        assert set(result["fields"]) == {"a", "b", "c"}

    def test_joins_lines_and_averages_confidence(self):
        ocr_fn = _stub_ocr(default=[("Nimal", 0.8), ("Perera", 0.6)])
        result = te.extract_fields(
            _textured(100, 100), _template(), ocr_fn=ocr_fn, align=False
        )
        assert result["fields"]["top"]["value"] == "Nimal Perera"
        assert result["fields"]["top"]["confidence"] == 0.7

    def test_skips_fields_with_no_text(self):
        template = _template(
            {"filled": {"box": [0.0, 0.0, 1.0, 0.5]}, "blank": {"box": [0.0, 0.5, 1.0, 0.5]}}
        )
        ocr_fn = _stub_ocr(mapping={0: [("12345", 0.9)], 1: []})

        result = te.extract_fields(
            _textured(100, 100), template, ocr_fn=ocr_fn, align=False
        )

        assert "blank" not in result["fields"]
        assert result["fields"]["filled"]["value"] == "12345"

    def test_whitespace_only_text_is_dropped(self):
        ocr_fn = _stub_ocr(default=[("   ", 0.9)])
        result = te.extract_fields(
            _textured(100, 100), _template(), ocr_fn=ocr_fn, align=False
        )
        assert result["fields"] == {}
        assert result["mean_confidence"] == 0.0

    def test_output_shape_matches_parse_fields(self):
        """The two strategies must be interchangeable for callers."""
        result = te.extract_fields(
            _textured(100, 100), _template(), ocr_fn=_stub_ocr(), align=False
        )
        assert set(result) == {"raw_text", "mean_confidence", "fields", "alignment"}
        assert result["raw_text"] == "top: value"

    def test_alignment_skipped_without_a_reference(self):
        result = te.extract_fields(
            _textured(100, 100), _template(), ocr_fn=_stub_ocr(), align=True
        )
        assert result["alignment"]["aligned"] is False
        assert result["alignment"]["reason"] == "alignment skipped"

    def test_aligns_when_a_reference_is_given(self):
        reference = _textured(320, 240)
        result = te.extract_fields(
            reference.copy(),
            _template(),
            ocr_fn=_stub_ocr(),
            reference=reference,
            align=True,
        )
        assert result["alignment"]["aligned"] is True

    def test_rotation_is_applied_before_cropping(self):
        """A box written in the upright frame must land on the rotated content.

        The scan is a landscape image whose left half is white; rotating it 90
        counter-clockwise puts that half at the bottom, so the template's top box
        should see black.
        """
        landscape = np.zeros((100, 200, 3), dtype=np.uint8)
        landscape[:, :100] = 255

        ocr_fn = _stub_ocr()
        template = _template()
        template["rotate"] = 90
        te.extract_fields(landscape, template, ocr_fn=ocr_fn, align=False)

        crop = ocr_fn.calls[0]
        assert crop.shape[0] > crop.shape[1]  # portrait after the turn
        assert crop.mean() < 10  # the white half rotated away from the top box

    def test_rotation_is_applied_to_the_reference_too(self):
        """Both frames must turn, or alignment compares different orientations."""
        reference = _textured(320, 240)
        template = _template()
        template["rotate"] = 90

        result = te.extract_fields(
            reference.copy(),
            template,
            ocr_fn=_stub_ocr(),
            reference=reference,
            align=True,
        )
        assert result["alignment"]["aligned"] is True

    def test_invalid_rotation_is_rejected_at_extraction(self):
        template = _template()
        template["rotate"] = 30
        with pytest.raises(te.TemplateError, match="counter-clockwise degrees"):
            te.extract_fields(
                _textured(100, 100), template, ocr_fn=_stub_ocr(), align=False
            )


class TestExtractFromPath:
    def test_reads_image_and_reference(self, tmp_path):
        reference = _textured(320, 240)
        _write_image(tmp_path / "reference.png", reference)
        scan = _write_image(tmp_path / "scan.png", reference.copy())

        template = _template()
        template["reference_image"] = "reference.png"
        ocr_fn = _stub_ocr()

        result = te.extract_from_path(
            scan, template, ocr_fn=ocr_fn, template_dir=tmp_path
        )

        assert result["alignment"]["aligned"] is True
        assert result["fields"]["top"]["value"] == "value"

    def test_missing_reference_degrades_gracefully(self, tmp_path):
        scan = _write_image(tmp_path / "scan.png", _textured(200, 200))
        template = _template()
        template["reference_image"] = "does-not-exist.png"

        result = te.extract_from_path(scan, template, ocr_fn=_stub_ocr())

        assert result["alignment"]["aligned"] is False
        assert result["fields"]["top"]["value"] == "value"
