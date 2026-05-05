"""Tests for :func:`zoedepth.laser.face_relief.auto_orient_to_face`.

The detector itself is exercised by ``test_face_relief.py``; here we
focus on the rotation contract: with stub landmarks indicating a known
roll angle, the rotation must roll the image by that angle.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PIL import Image

from zoedepth.laser import face_relief as fr
from zoedepth.laser.face_relief import FaceLandmarks, auto_orient_to_face


def _stub_landmarks(angle_deg: float, w: int = 256, h: int = 256) -> FaceLandmarks:
    """Place 478 landmarks; eyes positioned to imply a given roll angle."""
    pts = np.full((478, 2), (w / 2.0, h / 2.0), dtype=np.float32)
    # Anchor cheek pair so face_width_px is non-zero.
    pts[234] = (40.0, h / 2.0)
    pts[454] = (w - 40.0, h / 2.0)
    # Eyes: 100 px apart, tilted by `angle_deg` around the centre.
    half = 50.0
    rad = np.radians(angle_deg)
    cx, cy = w / 2.0, h / 2.0
    # Right eye outer (idx 33), left eye outer (idx 263).
    pts[33] = (cx - half * np.cos(rad), cy - half * np.sin(rad))
    pts[263] = (cx + half * np.cos(rad), cy + half * np.sin(rad))
    return FaceLandmarks(points=pts, width=w, height=h)


def test_auto_orient_no_face_returns_input_unchanged():
    img = Image.new("RGB", (128, 128), (128, 128, 128))
    with patch.object(fr, "detect_face_landmarks", return_value=None):
        out = auto_orient_to_face(img)
    assert out is img


def test_auto_orient_below_min_threshold_returns_input():
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    landmarks = _stub_landmarks(0.5)  # 0.5° tilt → below default 1.5° floor
    with patch.object(fr, "detect_face_landmarks", return_value=landmarks):
        out = auto_orient_to_face(img)
    # Same size, same content (no rotation applied).
    assert out.size == img.size
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_auto_orient_above_max_threshold_returns_input():
    """Suspect detection (e.g. 60° tilt) must be ignored."""
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    landmarks = _stub_landmarks(60.0)
    with patch.object(fr, "detect_face_landmarks", return_value=landmarks):
        out = auto_orient_to_face(img)
    assert out.size == img.size


def test_auto_orient_rotates_when_within_thresholds():
    """A 10° tilt should produce a rotated canvas different from the input."""
    img = Image.new("RGB", (256, 256), (200, 50, 50))
    # Mark a recognisable corner so we can confirm rotation happened.
    img.paste(Image.new("RGB", (16, 16), (0, 0, 0)), (0, 0))
    landmarks = _stub_landmarks(10.0)
    with patch.object(fr, "detect_face_landmarks", return_value=landmarks):
        out = auto_orient_to_face(img)
    # Rotation with expand=True changes the canvas size.
    assert out.size != img.size or not np.array_equal(np.asarray(out), np.asarray(img))


def test_auto_orient_handles_truncated_landmarks():
    """If the iris-aware refinement was skipped (only 468 landmarks), we still
    have eye outer corners — auto-orient must still work."""
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    landmarks = _stub_landmarks(8.0)
    truncated = FaceLandmarks(
        points=landmarks.points[:468], width=landmarks.width, height=landmarks.height,
    )
    with patch.object(fr, "detect_face_landmarks", return_value=truncated):
        out = auto_orient_to_face(img)
    # Either rotated (different size) or unchanged — never crashes.
    assert isinstance(out, Image.Image)
