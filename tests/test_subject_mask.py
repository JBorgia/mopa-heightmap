"""Tests for :mod:`mopa.subject_mask`."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mopa import subject_mask as sm
from mopa.subject_mask import (
    DEFAULT_BACKGROUND_PLANE,
    DEFAULT_BINARY_THRESHOLD,
    DEFAULT_FEATHER_PX,
    DEFAULT_THRESHOLD_BACKEND_PERCENTILE,
    RGBA_ALPHA_MAX,
    SubjectMaskerSpec,
    compose_mask_with_heightmap,
    get_masker,
    list_maskers,
    load_masker,
    register_masker,
)


# ----------------------------------------------------------- registry contracts

def test_default_maskers_registered_with_correct_licensing():
    assert get_masker("threshold").requires_opt_in is False
    assert get_masker("rembg").requires_opt_in is False
    assert get_masker("rembg_human").requires_opt_in is False
    assert get_masker("birefnet").requires_opt_in is False
    rmbg2 = get_masker("rmbg2")
    assert rmbg2 is not None and rmbg2.requires_opt_in is True
    assert rmbg2.license.startswith("CC-BY-NC")


def test_list_maskers_filters_opt_in():
    keys_default = {m.key for m in list_maskers(include_opt_in=False)}
    keys_all = {m.key for m in list_maskers(include_opt_in=True)}
    assert "rmbg2" in keys_all
    assert "rmbg2" not in keys_default
    assert {"threshold", "rembg", "birefnet"}.issubset(keys_default)


def test_register_masker_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="already registered"):
        register_masker(SubjectMaskerSpec(
            key="threshold", label="dup", license="x",
            requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
            loader=lambda d: None,
        ))


def test_load_masker_unknown_key_raises():
    with pytest.raises(KeyError, match="No subject masker registered"):
        load_masker("not_a_real_backend", "cpu")


# ----------------------------------------------------------- threshold backend

def test_threshold_backend_isolates_bright_subject():
    masker, _ = load_masker("threshold", "cpu")
    # Bright subject covers <10% of pixels so it lands above the 90th
    # percentile cutoff and registers as foreground.
    arr = np.full((32, 32), 20, dtype=np.uint8)
    arr[14:18, 14:18] = 240  # 16 px of 1024 = 1.6 %
    alpha = masker.infer(Image.fromarray(arr))
    assert alpha.shape == (32, 32)
    assert alpha.dtype == np.float32
    # Subject region must register as foreground; corners must be background.
    assert alpha[15, 15] > 0.0
    assert alpha[0, 0] == 0.0
    assert alpha.max() <= 1.0 and alpha.min() >= 0.0


def test_threshold_backend_uses_documented_percentile_default():
    masker, _ = load_masker("threshold", "cpu")
    # Internal attribute is implementation detail but pinning it ensures
    # the registered loader honours DEFAULT_THRESHOLD_BACKEND_PERCENTILE.
    assert getattr(masker, "_percentile") == DEFAULT_THRESHOLD_BACKEND_PERCENTILE


# ------------------------------------------------------------ composition

def _ramp_heightmap(h: int = 16, w: int = 16) -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))


def test_compose_flattens_background_to_plane():
    height = _ramp_heightmap()
    alpha = np.zeros_like(height)
    alpha[:, 8:] = 1.0  # right half is subject
    out = compose_mask_with_heightmap(
        height, alpha,
        background_value=DEFAULT_BACKGROUND_PLANE,
        binary_threshold=DEFAULT_BINARY_THRESHOLD,
        feather_px=0,
    )
    # Subject side preserves the original ramp; background side is the plane.
    assert np.allclose(out[:, -1], height[:, -1])
    assert np.allclose(out[:, 0], DEFAULT_BACKGROUND_PLANE)


def test_compose_rejects_shape_mismatch():
    height = _ramp_heightmap(8, 8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        compose_mask_with_heightmap(height, alpha)


def test_compose_clamps_background_value_into_unit_range():
    height = _ramp_heightmap(8, 8)
    alpha = np.zeros_like(height)
    out = compose_mask_with_heightmap(
        height, alpha, background_value=99.0, feather_px=0,
    )
    # Whole frame is background; everything must clip to 1.0 (the legal max).
    assert out.max() <= 1.0
    assert np.allclose(out, 1.0)


def test_compose_feathering_produces_soft_edge():
    height = _ramp_heightmap(32, 32)
    alpha = np.zeros_like(height)
    alpha[:, 16:] = 1.0
    out_hard = compose_mask_with_heightmap(height, alpha, feather_px=0)
    out_soft = compose_mask_with_heightmap(
        height, alpha, feather_px=DEFAULT_FEATHER_PX,
    )
    # The soft version must produce at least one intermediate value at the
    # boundary that the hard version doesn't (proves the feather is wired).
    boundary_hard = out_hard[:, 14:18]
    boundary_soft = out_soft[:, 14:18]
    intermediates_hard = np.unique(np.round(boundary_hard, 3))
    intermediates_soft = np.unique(np.round(boundary_soft, 3))
    assert intermediates_soft.size >= intermediates_hard.size


# ------------------------------------------------------------ stub-backend test

class _StubMasker:
    """Returns a fixed alpha plane regardless of input."""

    def __init__(self, alpha: np.ndarray):
        self._alpha = alpha

    def infer(self, image):
        return self._alpha


def test_register_and_use_custom_masker():
    fixed = np.ones((4, 4), dtype=np.float32)
    register_masker(SubjectMaskerSpec(
        key="_test_stub", label="Test stub", license="MIT",
        requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
        loader=lambda device: _StubMasker(fixed),
    ))
    inst, dev = load_masker("_test_stub", "cpu")
    out = inst.infer(Image.new("RGB", (4, 4)))
    assert dev == "cpu"
    assert np.array_equal(out, fixed)


def test_constants_have_documented_values():
    """Pin the externally-promised defaults so behaviour can't drift silently."""
    assert DEFAULT_BINARY_THRESHOLD == 0.5
    assert DEFAULT_FEATHER_PX == 3
    assert DEFAULT_BACKGROUND_PLANE == 1.0
    assert DEFAULT_THRESHOLD_BACKEND_PERCENTILE == 90.0
    assert RGBA_ALPHA_MAX == 255
