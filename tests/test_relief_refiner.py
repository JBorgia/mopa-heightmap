"""Tests for the bas-relief refiner registry + guided filter."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoedepth.laser.relief_refiner import (
    DEFAULT_GUIDED_FILTER_EPS,
    DEFAULT_GUIDED_FILTER_RADIUS,
    DEFAULT_REFINE_STRENGTH,
    DEFAULT_REFINER_KEY,
    REFINE_STRENGTH_MAX,
    REFINE_STRENGTH_MIN,
    ReliefRefinerSpec,
    get_refiner,
    guided_filter,
    list_refiners,
    load_refiner,
    register_refiner,
)


def test_constants_have_documented_values():
    assert DEFAULT_REFINER_KEY == "guided-filter"
    assert DEFAULT_REFINE_STRENGTH == 0.5
    assert REFINE_STRENGTH_MIN == 0.0
    assert REFINE_STRENGTH_MAX == 1.0
    assert DEFAULT_GUIDED_FILTER_RADIUS == 8
    assert DEFAULT_GUIDED_FILTER_EPS == 1e-3


def test_default_refiners_registered():
    keys = {s.key for s in list_refiners()}
    assert {"guided-filter", "controlnet-depth-bas-relief"} <= keys


def test_guided_filter_permissive_controlnet_opt_in():
    assert get_refiner("guided-filter").requires_opt_in is False
    assert get_refiner("controlnet-depth-bas-relief").requires_opt_in is True


def test_register_duplicate_raises():
    spec = ReliefRefinerSpec(
        key="guided-filter", label="x", license="MIT",
        requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
        loader=lambda d: object(),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_refiner(spec)


def test_load_unknown_raises():
    with pytest.raises(KeyError):
        load_refiner("not-real", "cpu")


def test_guided_filter_constant_input_returns_constant():
    g = np.full((16, 16), 0.5, dtype=np.float32)
    s = np.full((16, 16), 0.3, dtype=np.float32)
    out = guided_filter(g, s, radius=2, eps=1e-3)
    assert np.allclose(out, 0.3, atol=1e-3)


def test_guided_filter_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        guided_filter(np.zeros((8, 8)), np.zeros((4, 4)))


def test_guided_filter_rejects_non_2d_guide():
    with pytest.raises(ValueError, match="2-D"):
        guided_filter(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))


def test_guided_filter_smooths_noisy_input_along_guide_edges():
    g = np.zeros((16, 16), dtype=np.float32)
    g[:, 8:] = 1.0
    rng = np.random.default_rng(0)
    s = g + rng.normal(scale=0.1, size=g.shape).astype(np.float32)
    out = guided_filter(g, s, radius=2, eps=1e-3)
    # Output should be much closer to the clean edge than the noisy input.
    assert np.linalg.norm(out - g) < np.linalg.norm(s - g)


def test_refiner_blends_with_strength_zero_returns_input_heightmap():
    refiner, _ = load_refiner("guided-filter", "cpu")
    img = Image.new("L", (16, 16), color=128).convert("RGB")
    h = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
    out = refiner.refine(img, h, strength=REFINE_STRENGTH_MIN)
    assert np.allclose(out, h, atol=1e-6)


def test_refiner_blends_with_strength_one_changes_heightmap():
    refiner, _ = load_refiner("guided-filter", "cpu")
    img = Image.new("L", (16, 16), color=128).convert("RGB")
    rng = np.random.default_rng(1)
    h = rng.random((16, 16)).astype(np.float32)
    out = refiner.refine(img, h, strength=REFINE_STRENGTH_MAX)
    assert out.shape == h.shape
    assert not np.allclose(out, h)


def test_refiner_rejects_non_2d_heightmap():
    refiner, _ = load_refiner("guided-filter", "cpu")
    img = Image.new("L", (8, 8)).convert("RGB")
    with pytest.raises(ValueError, match="2-D"):
        refiner.refine(img, np.zeros((8, 8, 3), dtype=np.float32))


def test_controlnet_loader_raises_when_package_absent():
    # diffusers may or may not be installed; if it IS, the stub still raises
    # the "not wired" error at .refine() time, not at load time.
    try:
        refiner, _ = load_refiner("controlnet-depth-bas-relief", "cpu")
    except RuntimeError as exc:
        assert "ControlNet" in str(exc)
        return
    with pytest.raises(RuntimeError, match="not wired"):
        refiner.refine(
            Image.new("RGB", (8, 8)), np.zeros((8, 8), dtype=np.float32),
        )
