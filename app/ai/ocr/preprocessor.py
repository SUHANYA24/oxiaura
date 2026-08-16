"""Image preprocessing for OCR (BUILD_SPEC Phase 5).

A small OpenCV pipeline that improves EasyOCR accuracy on photographed /
scanned documents: grayscale -> denoise -> deskew -> adaptive threshold ->
upscale. Each step is a standalone function so it can be tuned or tested in
isolation; ``preprocess`` runs them in order and returns a single-channel
``uint8`` image ready to hand to the extractor.
"""

from __future__ import annotations

import cv2
import numpy as np

# Documents smaller than this on their long edge are upscaled — small text is
# the most common cause of poor OCR recall.
_MIN_LONG_EDGE = 1000


def load_image(file_path: str) -> np.ndarray:
    """Read an image from disk as a BGR array.

    :raises ValueError: if the file cannot be decoded as an image.
    """
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {file_path!r}.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise; falls back to a light blur on tiny images."""
    if min(gray.shape[:2]) < 20:
        return cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate the text skew angle and rotate the image upright.

    Uses the minimum-area rectangle around the dark (text) pixels. Returns the
    input unchanged when there is nothing to measure or the estimate is tiny.
    """
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 10:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV reports the angle in (-90, 0]; normalize to a small correction.
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return gray

    (h, w) = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize with a locally-adaptive threshold to handle uneven lighting."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )


def resize(gray: np.ndarray, min_long_edge: int = _MIN_LONG_EDGE) -> np.ndarray:
    """Upscale small images so text is large enough for reliable recognition."""
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge >= min_long_edge or long_edge == 0:
        return gray
    scale = min_long_edge / long_edge
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess(file_path: str) -> np.ndarray:
    """Run the full pipeline on the image at ``file_path``.

    :returns: a single-channel ``uint8`` image suitable for EasyOCR.
    """
    gray = to_grayscale(load_image(file_path))
    gray = denoise(gray)
    gray = deskew(gray)
    gray = resize(gray)
    return adaptive_threshold(gray)
