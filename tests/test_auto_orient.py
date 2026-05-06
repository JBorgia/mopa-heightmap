"""Tests for auto-orient face and auto-crop helpers.

These tests don't require a real face — they validate the no-face,
mediapipe-missing, and basic-cropping fallback paths so the module
behaves correctly when called on synthetic images.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mopa.imgproc.auto_orient import auto_orient_to_face, find_face_eye_angle
from mopa.imgproc.auto_crop import (
    auto_crop_to_aspect,
    find_face_bbox,
    find_saliency_centre,
)


def _flat_image(w: int = 64, h: int = 48, color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# ----------------------------------------------------------- auto-orient

def test_auto_orient_no_face_returns_image_unchanged():
    """A synthetic flat photo has no face — should be a clean no-op."""
    img = _flat_image(96, 72)
    out, angle = auto_orient_to_face(img)
    assert angle == 0.0
    assert out.size == img.size
    # Same bytes (no rotation applied).
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_auto_orient_below_threshold_no_op():
    """Even if eye angle were detected as 0.5°, the threshold blocks rotation."""
    img = _flat_image(96, 72)
    out, angle = auto_orient_to_face(img, min_angle_deg=10.0)
    assert angle == 0.0


def test_find_face_eye_angle_returns_none_for_flat_photo():
    img = _flat_image(48, 48)
    assert find_face_eye_angle(img) is None


# ----------------------------------------------------------- auto-crop

def test_find_face_bbox_returns_none_for_flat_photo():
    img = _flat_image(64, 64)
    assert find_face_bbox(img) is None


def test_find_saliency_centre_returns_near_centre_for_flat_photo():
    img = _flat_image(64, 48)
    cx, cy = find_saliency_centre(img)
    # Saliency is degenerate on flat input; result should land within
    # a couple pixels of the image centre (DCT-spectral can drift by 1
    # pixel on artifacts at the borders).
    assert abs(cx - 32) <= 2
    assert abs(cy - 24) <= 2


def test_find_saliency_centre_finds_bright_blob():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:20, 40:50] = 255  # bright square in upper-right
    img = Image.fromarray(arr, "RGB")
    cx, cy = find_saliency_centre(img)
    # Should be biased toward the bright region.
    assert cx > 32  # rightward bias
    assert cy < 32  # upward bias


def test_auto_crop_to_aspect_square_target():
    img = _flat_image(96, 64)
    out, strategy = auto_crop_to_aspect(img, target_aspect=1.0)
    assert strategy in {"face", "saliency", "center"}
    # Output should be square (within rounding).
    out_w, out_h = out.size
    assert abs(out_w - out_h) <= 1


def test_auto_crop_to_aspect_landscape_target():
    img = _flat_image(64, 64)
    out, _ = auto_crop_to_aspect(img, target_aspect=2.0)
    out_w, out_h = out.size
    # 2:1 aspect, height-limited from a 1:1 input.
    assert pytest.approx(out_w / out_h, rel=0.05) == 2.0


def test_auto_crop_to_aspect_portrait_target():
    img = _flat_image(64, 64)
    out, _ = auto_crop_to_aspect(img, target_aspect=0.5)
    out_w, out_h = out.size
    assert pytest.approx(out_w / out_h, rel=0.05) == 0.5
