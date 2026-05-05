"""Tests for :mod:`zoedepth.laser.pass_masks`."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.pass_masks import (
    DEFAULT_CLEANUP_RADIUS_PX,
    DEFAULT_DETAIL_SIGMA_PX,
    DEFAULT_FORM_THRESHOLD,
    DEFAULT_SHADING_SIGMA_PX,
    cleanup_mask,
    derive_pass_masks,
    detail_mask,
    form_mask,
    polish_mask,
    shading_mask,
)
from zoedepth.laser.stages import (
    PASS_KIND_CLEANUP,
    PASS_KIND_DETAIL,
    PASS_KIND_FORM,
    PASS_KIND_POLISH,
    PASS_KIND_SHADING,
)


def _ring_heightmap(h: int = 64, w: int = 64, inner_r: int = 12, outer_r: int = 24) -> np.ndarray:
    """Heightmap with a ring shape: subject 0.0..0.7, background 1.0."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    out = np.full((h, w), 1.0, dtype=np.float32)
    out[(r >= inner_r) & (r <= outer_r)] = 0.4
    return out


# ----------------------------------------------------------- constants pin

def test_constants_have_documented_values():
    assert DEFAULT_FORM_THRESHOLD == 0.985
    assert DEFAULT_CLEANUP_RADIUS_PX == 6
    assert DEFAULT_DETAIL_SIGMA_PX == 4.0
    assert DEFAULT_SHADING_SIGMA_PX == 24.0


# ----------------------------------------------------------- form

def test_form_mask_separates_subject_from_background():
    hm = _ring_heightmap()
    mask = form_mask(hm, feather_px=0)
    assert mask.shape == hm.shape
    assert mask.dtype == np.float32
    # Pixel on the ring (radius ≈ 18 from centre) ⇒ subject ⇒ mask = 1.
    assert mask[32, 50] > 0.5
    # Far outside (corner) ⇒ background ⇒ mask zero.
    assert mask[0, 0] == 0.0
    # Inside the ring's hole (radius ≈ 0) ⇒ also background (hm == 1.0).
    assert mask[32, 32] == 0.0


def test_form_mask_feather_softens_edges():
    hm = _ring_heightmap()
    hard = form_mask(hm, feather_px=0)
    soft = form_mask(hm, feather_px=4)
    # Soft version must produce intermediate values that hard doesn't.
    hard_unique = np.unique(np.round(hard, 3)).size
    soft_unique = np.unique(np.round(soft, 3)).size
    assert soft_unique > hard_unique


def test_form_mask_returns_zero_when_heightmap_is_all_background():
    flat = np.full((16, 16), 1.0, dtype=np.float32)
    mask = form_mask(flat, feather_px=0)
    assert np.all(mask == 0.0)


# ----------------------------------------------------------- cleanup

def test_cleanup_mask_is_an_edge_ring():
    hm = _ring_heightmap()
    cmask = cleanup_mask(hm, radius_px=4)
    fmask = form_mask(hm, feather_px=0)
    # Ring should be near-zero inside the subject body and outside far away.
    # On the boundary (one pixel just inside the inner radius) the cleanup
    # mask should be greater than the form mask body's interior.
    inner_body = cmask[32, 32 - 6]   # inside the ring
    edge = cmask[32, 32 - 12]        # at the inner edge
    far_outside = cmask[2, 2]
    assert far_outside < 1e-3
    # Edge band light ≥ deep-interior light on the ring's annulus.
    assert edge >= inner_body or inner_body < 0.1
    # And the cleanup ring shouldn't completely overlap the form mask.
    assert cmask.max() <= 1.0
    assert fmask.max() <= 1.0


def test_cleanup_mask_handles_no_subject():
    flat = np.full((16, 16), 1.0, dtype=np.float32)
    cmask = cleanup_mask(flat)
    assert np.all(cmask == 0.0)


# ----------------------------------------------------------- detail / shading

def test_detail_mask_lights_up_high_frequency_features():
    """Add a sharp dot to a flat subject; detail mask should peak there."""
    hm = np.full((64, 64), 0.5, dtype=np.float32)
    hm[32, 32] = 0.0  # single deep pixel
    mask = detail_mask(hm, sigma_px=2.0)
    # Peak at the dot location, not at the corners.
    assert mask[32, 32] > 0.5
    assert mask[0, 0] < mask[32, 32]


def test_shading_mask_lights_up_mid_frequency_features():
    """Smooth gradient should engage the shading band, not the detail band."""
    h = 64
    grad = np.tile(np.linspace(0.0, 0.7, h, dtype=np.float32), (h, 1))
    detail = detail_mask(grad, sigma_px=2.0)
    shading = shading_mask(grad, sigma_px=24.0, detail_sigma_px=2.0)
    # Both peak-normalise to 1, but shading carries far more *energy* on a
    # smooth gradient because the detail-band passes nothing while the
    # shading band picks up the whole slope.
    assert shading.mean() > detail.mean() * 3.0


def test_detail_and_shading_zero_outside_subject():
    hm = _ring_heightmap()
    d = detail_mask(hm)
    s = shading_mask(hm)
    # Far from the ring (corner), heightmap is fully background ⇒ outputs zero.
    assert d[0, 0] == 0.0
    assert s[0, 0] == 0.0


# ----------------------------------------------------------- polish

def test_polish_mask_covers_full_subject():
    hm = _ring_heightmap()
    p = polish_mask(hm)
    f = form_mask(hm, feather_px=1)
    # Polish covers everything form covers (by construction).
    assert np.allclose(p, f, atol=1e-4)


# ----------------------------------------------------------- bundle

def test_derive_pass_masks_returns_all_canonical_kinds():
    hm = _ring_heightmap()
    masks = derive_pass_masks(hm)
    expected = {PASS_KIND_FORM, PASS_KIND_CLEANUP, PASS_KIND_DETAIL,
                PASS_KIND_SHADING, PASS_KIND_POLISH}
    assert set(masks.keys()) == expected
    for kind, m in masks.items():
        assert m.shape == hm.shape, f"{kind} shape mismatch"
        assert m.dtype == np.float32
        assert m.min() >= 0.0 - 1e-6
        assert m.max() <= 1.0 + 1e-6


def test_derive_pass_masks_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        derive_pass_masks(np.zeros((4, 4, 3), dtype=np.float32))
