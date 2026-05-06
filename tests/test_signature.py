"""Tests for :mod:`mopa.signature`."""
from __future__ import annotations

import numpy as np
import pytest

from mopa.signature import (
    DEFAULT_CORNER,
    DEFAULT_HEIGHT_FRACTION,
    DEFAULT_MARGIN_FRACTION,
    VALID_CORNERS,
    render_text_signature_mask,
)


# ----------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert DEFAULT_CORNER == "br"
    assert DEFAULT_HEIGHT_FRACTION == 0.04
    assert DEFAULT_MARGIN_FRACTION == 0.03
    assert VALID_CORNERS == ("tl", "tr", "bl", "br")


# ----------------------------------------------------------- shape contracts

def test_render_returns_correct_shape_and_dtype():
    mask = render_text_signature_mask((128, 256), "JB 2026")
    assert mask.shape == (128, 256)
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() and mask.max() <= 1.0


def test_empty_text_returns_zero_mask():
    mask = render_text_signature_mask((64, 64), "")
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert np.all(mask == 0.0)


def test_invalid_corner_raises():
    with pytest.raises(ValueError, match="corner must be"):
        render_text_signature_mask((64, 64), "X", corner="middle")


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        render_text_signature_mask((0, 32), "X")
    with pytest.raises(ValueError, match="shape"):
        render_text_signature_mask((32, -1), "X")


# ----------------------------------------------------------- placement

def _half_quadrants(mask: np.ndarray) -> dict[str, float]:
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    return {
        "tl": float(mask[:cy, :cx].sum()),
        "tr": float(mask[:cy, cx:].sum()),
        "bl": float(mask[cy:, :cx].sum()),
        "br": float(mask[cy:, cx:].sum()),
    }


@pytest.mark.parametrize("corner", VALID_CORNERS)
def test_text_lands_in_requested_corner(corner: str):
    mask = render_text_signature_mask(
        (256, 256), "TEST", corner=corner, height_fraction=0.08,
    )
    quadrants = _half_quadrants(mask)
    # The chosen corner must hold the most ink.
    chosen_mass = quadrants[corner]
    other_max = max(v for k, v in quadrants.items() if k != corner)
    assert chosen_mass > other_max, (
        f"corner {corner} has {chosen_mass} ink, others up to {other_max}"
    )


# ----------------------------------------------------------- custom mask

def test_custom_mask_is_used_in_place_of_text():
    custom = np.zeros((32, 32), dtype=np.float32)
    custom[:16, :] = 1.0
    out = render_text_signature_mask((32, 32), "ignored", custom_mask=custom)
    assert np.allclose(out, custom)


def test_custom_mask_resampled_to_shape():
    custom = np.ones((16, 16), dtype=np.float32)
    out = render_text_signature_mask((64, 64), "", custom_mask=custom)
    assert out.shape == (64, 64)


def test_custom_mask_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        render_text_signature_mask((32, 32), "x", custom_mask=np.zeros((4, 4, 3)))
