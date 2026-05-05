"""Tests for multi-resolution depth fusion."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoedepth.laser.depth_fusion import (
    DEFAULT_FUSION_HIGHPASS_WEIGHT,
    DEFAULT_FUSION_SCALES,
    DEFAULT_LOWPASS_SIGMA_PX,
    MIN_FUSION_SCALES,
    MultiResolutionDepth,
    fuse_depths,
)


def test_constants_have_documented_values():
    assert DEFAULT_FUSION_SCALES == (512, 1024, 2048)
    assert DEFAULT_LOWPASS_SIGMA_PX == 24.0
    assert DEFAULT_FUSION_HIGHPASS_WEIGHT == 1.0
    assert MIN_FUSION_SCALES == 1


def test_fuse_depths_requires_at_least_one_input():
    with pytest.raises(ValueError, match="at least"):
        fuse_depths([])


def test_fuse_depths_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        fuse_depths([np.zeros((4, 4, 3), dtype=np.float32)])


def test_fuse_depths_single_input_returns_blurred_self():
    rng = np.random.default_rng(0)
    d = rng.random((32, 32)).astype(np.float32)
    out = fuse_depths([d], sigma_px=2.0)
    assert out.shape == d.shape
    assert out.dtype == np.float32
    # With a single input the highpass loop never runs, so output is the
    # blurred low-frequency band only.
    assert out.std() < d.std()


def test_fuse_depths_resamples_to_largest_input_shape():
    small = np.zeros((8, 8), dtype=np.float32)
    big = np.zeros((32, 32), dtype=np.float32)
    out = fuse_depths([small, big], sigma_px=1.0)
    assert out.shape == big.shape


def test_fuse_depths_recovers_constant_field():
    a = np.full((16, 16), 0.5, dtype=np.float32)
    b = np.full((32, 32), 0.5, dtype=np.float32)
    out = fuse_depths([a, b], sigma_px=2.0)
    assert np.allclose(out, 0.5, atol=1e-3)


def test_fuse_depths_preserves_high_frequency_when_weight_one():
    base = np.zeros((32, 32), dtype=np.float32)
    spike = np.zeros((32, 32), dtype=np.float32)
    spike[16, 16] = 1.0
    fused = fuse_depths([base, spike], sigma_px=2.0, highpass_weight=1.0)
    # High-freq spike survives the fusion at the output resolution.
    assert fused[16, 16] > 0.5


def test_fuse_depths_drops_high_frequency_when_weight_zero():
    base = np.zeros((32, 32), dtype=np.float32)
    spike = np.zeros((32, 32), dtype=np.float32)
    spike[16, 16] = 1.0
    fused = fuse_depths([base, spike], sigma_px=2.0, highpass_weight=0.0)
    # With no high-pass contribution, the spike vanishes.
    assert fused[16, 16] < 0.05


# --------------------------------------------------- MultiResolutionDepth

class _ConstantBackend:
    """Returns a constant-valued depth at whatever input resolution."""

    def __init__(self, value: float) -> None:
        self._value = float(value)
        self.calls: list[tuple[int, int]] = []

    def infer_pil(self, image: Image.Image, **_kwargs) -> np.ndarray:
        w, h = image.size
        self.calls.append((w, h))
        return np.full((h, w), self._value, dtype=np.float32)


def test_multi_resolution_calls_backend_at_each_scale():
    backend = _ConstantBackend(value=0.5)
    runner = MultiResolutionDepth(backend=backend, scales=(64, 128))
    img = Image.new("RGB", (256, 128))
    out = runner.infer_pil(img)
    assert len(backend.calls) == 2
    long_sides = [max(w, h) for w, h in backend.calls]
    assert sorted(long_sides) == [64, 128]
    # All inputs constant => output should be ~constant.
    assert np.allclose(out, 0.5, atol=1e-3)


def test_multi_resolution_rejects_empty_scales():
    runner = MultiResolutionDepth(backend=_ConstantBackend(0.0), scales=())
    with pytest.raises(ValueError, match="non-empty"):
        runner.infer_pil(Image.new("RGB", (32, 32)))


def test_multi_resolution_rejects_non_positive_scale():
    runner = MultiResolutionDepth(backend=_ConstantBackend(0.0), scales=(64, 0))
    with pytest.raises(ValueError, match="positive"):
        runner.infer_pil(Image.new("RGB", (32, 32)))
