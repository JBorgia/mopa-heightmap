"""Tests for :mod:`zoedepth.laser.pass_masks`."""
from __future__ import annotations

import numpy as np

from zoedepth.laser.pass_masks import (
    DEFAULT_PHOTO_TONAL_LEVELS,
    DEFAULT_PHOTO_TONAL_STRENGTH,
    photo_tonal_mask,
)


def _gradient_photo(h: int = 32, w: int = 64) -> np.ndarray:
    """RGB photo with a horizontal luma gradient (left=dark, right=bright)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    arr[:] = grad[None, :, None]
    return arr


# ----------------------------------------------------------- constants pin

def test_constants_have_documented_values():
    assert DEFAULT_PHOTO_TONAL_LEVELS == 32
    assert DEFAULT_PHOTO_TONAL_STRENGTH == 0.7


# ----------------------------------------------------------- polarity

def test_photo_tonal_default_polarity_carves_dark_regions():
    photo = _gradient_photo()
    mask = photo_tonal_mask(photo, subject_alpha=None, dither=False, strength=1.0)
    # Default invert=False: dark photo → high mask (laser fires more).
    assert mask[:, 0].mean() > mask[:, -1].mean()


def test_photo_tonal_invert_flips_polarity():
    photo = _gradient_photo()
    mask = photo_tonal_mask(photo, subject_alpha=None, dither=False, invert=True, strength=1.0)
    # invert=True: bright photo → high mask.
    assert mask[:, -1].mean() > mask[:, 0].mean()


def test_photo_tonal_subject_alpha_zeroes_background():
    photo = _gradient_photo()
    # Subject is the LEFT half (where the photo is dark → high mask under
    # the default invert=False polarity).
    alpha = np.zeros(photo.shape[:2], dtype=np.float32)
    alpha[:, : photo.shape[1] // 2] = 1.0
    mask = photo_tonal_mask(photo, subject_alpha=alpha, dither=False, strength=1.0)
    # Right half is outside the subject → mask zeroed even though the
    # photo is bright there.
    assert mask[:, -1].max() == 0.0
    # Left half retains the dark photo's high-engraving signal.
    assert mask[:, 0].max() > 0.0


def test_photo_tonal_strength_scales_output():
    photo = _gradient_photo()
    full = photo_tonal_mask(photo, subject_alpha=None, dither=False, strength=1.0)
    half = photo_tonal_mask(photo, subject_alpha=None, dither=False, strength=0.5)
    # half-strength should not exceed full-strength
    assert half.max() <= full.max() + 1e-5
    # peak should drop roughly with strength
    assert half.max() < full.max()


def test_photo_tonal_dither_changes_distribution():
    photo = _gradient_photo()
    smooth = photo_tonal_mask(photo, subject_alpha=None, dither=False, strength=1.0)
    dithered = photo_tonal_mask(
        photo, subject_alpha=None, dither=True, dither_levels=4, strength=1.0,
    )
    # Dithering quantises so unique values shrink.
    assert np.unique(np.round(dithered, 3)).size <= 6
    # Means should be close (dither preserves average intensity).
    assert abs(dithered.mean() - smooth.mean()) < 0.1


def test_photo_tonal_subject_alpha_shape_mismatch_raises():
    photo = _gradient_photo()
    bad = np.ones((photo.shape[0] + 5, photo.shape[1]), dtype=np.float32)
    try:
        photo_tonal_mask(photo, subject_alpha=bad, dither=False)
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on shape mismatch")
