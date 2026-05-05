"""Tests for :mod:`zoedepth.laser.face_relief`."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoedepth.laser import face_relief
from zoedepth.laser.face_relief import (
    DEFAULT_FACE_RELIEF_STRENGTH,
    DEFAULT_REGION_OFFSETS,
    FaceLandmarks,
    REGION_KEY_LANDMARKS,
    REGION_RADIUS_UNITS,
    apply_face_relief,
    build_region_masks,
    detect_face_landmarks,
)


# ----------------------------------------------------------- constants pin

def test_default_offsets_are_signed_and_subtle():
    # Sanity: deepening is negative, raising is positive, magnitudes ≤ 0.25.
    assert DEFAULT_REGION_OFFSETS["nostril_left"] < 0
    assert DEFAULT_REGION_OFFSETS["nose_tip"] > 0
    for offset in DEFAULT_REGION_OFFSETS.values():
        assert -0.25 <= offset <= 0.25


def test_every_region_has_landmarks_and_radius():
    for region in DEFAULT_REGION_OFFSETS:
        assert region in REGION_KEY_LANDMARKS, f"{region} missing key landmarks"
        assert region in REGION_RADIUS_UNITS, f"{region} missing radius"


# ----------------------------------------------------------- detector

def test_detect_face_landmarks_returns_none_on_tiny_image():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    assert detect_face_landmarks(img) is None


def test_detect_face_landmarks_returns_none_on_blank_image():
    # Solid grey: MediaPipe should find no face.
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    assert detect_face_landmarks(img) is None


# ----------------------------------------------------------- region masks

def _fake_landmarks(h: int = 64, w: int = 64) -> FaceLandmarks:
    """Place 478 landmarks in a stable, well-spaced grid for mask tests."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(low=[10.0, 10.0], high=[w - 10.0, h - 10.0],
                      size=(478, 2)).astype(np.float32)
    # Anchor cheek-pair (234 ↔ 454) so face_width_px is well-defined.
    pts[234] = (15.0, h / 2.0)
    pts[454] = (w - 15.0, h / 2.0)
    return FaceLandmarks(points=pts, width=w, height=h)


def test_build_region_masks_returns_one_per_region():
    landmarks = _fake_landmarks()
    masks = build_region_masks(landmarks, (64, 64))
    assert set(masks.keys()) == set(REGION_KEY_LANDMARKS.keys())
    for name, mask in masks.items():
        assert mask.shape == (64, 64), f"{name} shape mismatch"
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() and mask.max() <= 1.0 + 1e-6
        # Each region has at least one positive splat.
        assert mask.max() > 0.5, f"{name} has no peak"


def test_build_region_masks_skips_out_of_range_indices():
    """Iris indices 468/473 require refine_landmarks=True; truncate gracefully."""
    landmarks = _fake_landmarks()
    # Truncate to 468 so iris splats are out-of-range.
    landmarks_truncated = FaceLandmarks(
        points=landmarks.points[:468], width=landmarks.width, height=landmarks.height,
    )
    masks = build_region_masks(landmarks_truncated, (64, 64))
    # Iris masks remain in the dict but with no contributions = max ≤ floor.
    assert masks["iris_left"].max() <= 1e-6
    # Other regions still light up.
    assert masks["nose_tip"].max() > 0.5


# ----------------------------------------------------------- application

def test_apply_face_relief_no_op_when_strength_is_zero():
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    hm = np.full((64, 64), 0.5, dtype=np.float32)
    out = apply_face_relief(hm, img, strength=0.0)
    assert np.array_equal(out, hm)


def test_apply_face_relief_no_op_when_no_face_detected():
    # Solid grey image; MediaPipe should detect no face and the heightmap
    # must round-trip unchanged.
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    hm = np.full((256, 256), 0.5, dtype=np.float32)
    out = apply_face_relief(hm, img, strength=1.0)
    assert np.array_equal(out, hm)


def test_apply_face_relief_with_synthetic_landmarks_modifies_heightmap():
    """When a face is detected, the heightmap must change."""
    img = Image.new("RGB", (128, 128), (128, 128, 128))
    hm = np.full((128, 128), 0.5, dtype=np.float32)
    landmarks = _fake_landmarks(128, 128)
    out = apply_face_relief(hm, img, strength=1.0, landmarks=landmarks)
    assert out.shape == hm.shape
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.array_equal(out, hm), "expected heightmap to change"


def test_apply_face_relief_polarity_flips_for_white_is_deep():
    """Region offsets must invert sign when black_is_deep is False."""
    img = Image.new("RGB", (128, 128), (128, 128, 128))
    hm = np.full((128, 128), 0.5, dtype=np.float32)
    landmarks = _fake_landmarks(128, 128)
    bid = apply_face_relief(hm, img, strength=1.0, black_is_deep=True, landmarks=landmarks)
    wid = apply_face_relief(hm, img, strength=1.0, black_is_deep=False, landmarks=landmarks)
    # Around the input mean of 0.5, the per-pixel deltas should be opposite.
    delta_bid = bid - hm
    delta_wid = wid - hm
    # Sum of pixelwise dot products is negative when polarities are opposite.
    assert float((delta_bid * delta_wid).sum()) < 0.0


def test_apply_face_relief_clips_to_unit_range():
    """Output must remain in [0, 1] even for heightmaps already at the limits."""
    img = Image.new("RGB", (128, 128), (128, 128, 128))
    hm = np.full((128, 128), 0.99, dtype=np.float32)
    landmarks = _fake_landmarks(128, 128)
    out = apply_face_relief(hm, img, strength=1.5, landmarks=landmarks)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_default_strength_is_unity():
    # Pin the default so tests/profiles agree on what '1.0' means.
    assert DEFAULT_FACE_RELIEF_STRENGTH == 1.0
