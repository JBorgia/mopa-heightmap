"""Tests for :mod:`zoedepth.laser.external_heightmap`.

Covers polarity normalisation, auto-stretch into the engraving budget,
resolution alignment, and the end-to-end ``fit_external_heightmap_to_photo``
helper that ``service.render()`` uses when ``external_heightmap_path``
is set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zoedepth.laser.external_heightmap import (
    DEFAULT_AUTO_STRETCH,
    DEFAULT_POLARITY,
    EXTERNAL_DEPTH_DEEP_LIMIT,
    EXTERNAL_DEPTH_SURFACE_LIMIT,
    auto_stretch_subject,
    fit_external_heightmap_to_photo,
    load_external_heightmap,
    normalise_polarity,
)


def _save_synthetic_heightmap(
    path: Path, *, w: int = 64, h: int = 64,
    bright_raised: bool = True, bit_depth: int = 8,
) -> None:
    """Subject (centre disc) bright on dark bg by default."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    arr = np.where(r < 20, 0.85, 0.05).astype(np.float32)
    arr = arr - 0.05 * np.exp(-((yy - cy) ** 2 + (xx - cx + 5) ** 2) / 30.0)  # eye socket
    if not bright_raised:
        arr = 1.0 - arr
    if bit_depth == 16:
        Image.fromarray((arr * 65535).astype(np.uint16), mode="I;16").save(path)
    else:
        Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


# ----------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert DEFAULT_POLARITY == "bright_raised"
    assert DEFAULT_AUTO_STRETCH is True
    assert EXTERNAL_DEPTH_DEEP_LIMIT == 0.02
    assert EXTERNAL_DEPTH_SURFACE_LIMIT == 0.98


# ----------------------------------------------------------- polarity

def test_normalise_polarity_bright_raised_is_identity():
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = normalise_polarity(arr, "bright_raised")
    assert np.array_equal(out, arr)


def test_normalise_polarity_dark_raised_inverts():
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = normalise_polarity(arr, "dark_raised")
    assert np.allclose(out, 1.0 - arr)


def test_normalise_polarity_auto_picks_bright_raised_for_dark_corners():
    """Sculptok-style: bright subject on black background → bright_raised."""
    arr = np.full((32, 32), 0.05, dtype=np.float32)
    arr[10:22, 10:22] = 0.9
    out = normalise_polarity(arr, "auto")
    # bright_raised is identity, so the centre should still be bright.
    assert out[16, 16] > 0.5


def test_normalise_polarity_auto_picks_dark_raised_for_light_corners():
    arr = np.full((32, 32), 0.95, dtype=np.float32)
    arr[10:22, 10:22] = 0.1
    out = normalise_polarity(arr, "auto")
    # dark_raised inverts; centre was 0.1, becomes 0.9.
    assert out[16, 16] > 0.5


def test_normalise_polarity_rejects_unknown_string():
    with pytest.raises(ValueError, match="Unknown polarity"):
        normalise_polarity(np.zeros((4, 4), dtype=np.float32), "sideways")


# ----------------------------------------------------------- auto-stretch

def test_auto_stretch_no_mask_treats_full_frame_as_subject():
    arr = np.linspace(0.2, 0.7, 16, dtype=np.float32).reshape(4, 4)
    out = auto_stretch_subject(arr, None)
    # Output range fills [deep_limit, surface_limit].
    assert out.min() == pytest.approx(EXTERNAL_DEPTH_DEEP_LIMIT, abs=0.01)
    assert out.max() == pytest.approx(EXTERNAL_DEPTH_SURFACE_LIMIT, abs=0.01)


def test_auto_stretch_with_mask_flattens_background():
    arr = np.full((16, 16), 0.5, dtype=np.float32)
    arr[6:10, 6:10] = 0.9   # bright subject
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[6:10, 6:10] = 1.0
    out = auto_stretch_subject(arr, mask, background_value=1.0)
    # Background pixels untouched at 1.0; subject pixels live in
    # the engraving range.
    assert np.all(out[mask < 0.5] == 1.0)
    inside = out[mask >= 0.5]
    assert inside.min() >= EXTERNAL_DEPTH_DEEP_LIMIT - 1e-6
    assert inside.max() <= EXTERNAL_DEPTH_SURFACE_LIMIT + 1e-6


def test_auto_stretch_rejects_mask_shape_mismatch():
    arr = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        auto_stretch_subject(arr, mask)


def test_auto_stretch_handles_empty_subject():
    arr = np.full((8, 8), 0.5, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    out = auto_stretch_subject(arr, mask, background_value=1.0)
    # No subject pixels → entire output is background_value.
    assert np.all(out == 1.0)


# ----------------------------------------------------------- load

def test_load_external_heightmap_from_8bit_png(tmp_path: Path):
    p = tmp_path / "synthetic.png"
    _save_synthetic_heightmap(p, w=64, h=64, bright_raised=True)
    arr = load_external_heightmap(p)
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_load_external_heightmap_from_16bit_png(tmp_path: Path):
    p = tmp_path / "synthetic_16.png"
    _save_synthetic_heightmap(p, w=48, h=48, bit_depth=16)
    arr = load_external_heightmap(p)
    assert arr.shape == (48, 48)
    # 16-bit precision survives the round-trip with much finer levels
    # than an 8-bit load. Pixel-accurate spot check is overkill; just
    # confirm it loaded into the legal range.
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_load_external_heightmap_resizes_to_target(tmp_path: Path):
    p = tmp_path / "synthetic.png"
    _save_synthetic_heightmap(p, w=32, h=32)
    arr = load_external_heightmap(p, target_size=(96, 96))
    assert arr.shape == (96, 96)


def test_load_external_heightmap_dark_raised_inverts(tmp_path: Path):
    p = tmp_path / "synthetic_dark.png"
    _save_synthetic_heightmap(p, w=32, h=32, bright_raised=False)
    arr_bright = load_external_heightmap(p, polarity="bright_raised")
    arr_dark = load_external_heightmap(p, polarity="dark_raised")
    # The two should be inverses (within float-rounding) of each other.
    assert np.allclose(arr_bright + arr_dark, 1.0, atol=2.0 / 255.0)


def test_load_external_heightmap_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_external_heightmap(tmp_path / "does_not_exist.png")


# ----------------------------------------------------------- fit (end-to-end)

def test_fit_external_heightmap_resizes_and_stretches(tmp_path: Path):
    p = tmp_path / "synthetic.png"
    _save_synthetic_heightmap(p, w=32, h=32, bright_raised=True)
    photo_size = (96, 96)
    out = fit_external_heightmap_to_photo(
        p, photo_size, subject_alpha=None,
    )
    assert out.shape == (96, 96)
    # No mask supplied → full-frame stretched. Output must use the full
    # engraving budget.
    assert out.min() == pytest.approx(EXTERNAL_DEPTH_DEEP_LIMIT, abs=0.01)
    assert out.max() == pytest.approx(EXTERNAL_DEPTH_SURFACE_LIMIT, abs=0.01)


def test_fit_external_heightmap_with_subject_mask_flattens_bg(tmp_path: Path):
    p = tmp_path / "synthetic.png"
    _save_synthetic_heightmap(p, w=64, h=64, bright_raised=True)
    photo_size = (64, 64)
    # Subject mask: only the centre disc.
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    alpha = (r < 20).astype(np.float32)
    out = fit_external_heightmap_to_photo(
        p, photo_size, subject_alpha=alpha, background_value=1.0,
    )
    # Background is exactly the supplied background_value.
    assert np.all(out[alpha < 0.5] == pytest.approx(1.0))
    # Subject pixels live in the engraving range.
    inside = out[alpha >= 0.5]
    assert inside.min() >= EXTERNAL_DEPTH_DEEP_LIMIT - 1e-6
    assert inside.max() <= EXTERNAL_DEPTH_SURFACE_LIMIT + 1e-6


def test_fit_external_heightmap_no_stretch_preserves_input(tmp_path: Path):
    """When auto_stretch=False, the loaded heightmap survives unchanged
    (polarity-normalised, resized, but otherwise untouched). The output
    range matches the input's, NOT the engraving budget."""
    p = tmp_path / "synthetic.png"
    _save_synthetic_heightmap(p, w=32, h=32, bright_raised=True)
    raw = load_external_heightmap(p, target_size=(32, 32))
    out = fit_external_heightmap_to_photo(
        p, (32, 32), subject_alpha=None, auto_stretch=False,
    )
    assert out.shape == (32, 32)
    # Output is the loaded heightmap unmodified.
    assert np.allclose(out, raw, atol=1e-6)
