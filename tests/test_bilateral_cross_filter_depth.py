"""Tests for :func:`zoedepth.laser.heightmap.bilateral_cross_filter_depth`."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.heightmap import bilateral_cross_filter_depth


def test_no_guide_returns_input_unchanged():
    depth = np.linspace(0.0, 10.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = bilateral_cross_filter_depth(depth, None)
    assert np.allclose(out, depth)


def test_rejects_non_2d_depth():
    depth = np.zeros((4, 4, 3), dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D"):
        bilateral_cross_filter_depth(depth, rgb)


def test_returns_input_when_ximgproc_missing(monkeypatch):
    """If cv2.ximgproc isn't available, fall back to the unmodified depth."""
    import cv2

    if hasattr(cv2, "ximgproc"):
        monkeypatch.delattr(cv2, "ximgproc", raising=False)
    depth = np.linspace(0.0, 10.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    out = bilateral_cross_filter_depth(depth, rgb)
    assert out.dtype == np.float32
    assert np.allclose(out, depth)


def test_filter_preserves_scale_when_running():
    """Output depth must keep its absolute units regardless of normalisation."""
    import cv2

    if not hasattr(cv2, "ximgproc"):
        pytest.skip("cv2.ximgproc not available; filter falls back to no-op")

    rng = np.random.default_rng(0)
    depth = (rng.normal(loc=5.0, scale=2.0, size=(32, 32))).astype(np.float32)
    rgb = (rng.integers(0, 255, size=(32, 32, 3))).astype(np.uint8)
    out = bilateral_cross_filter_depth(depth, rgb)
    assert out.shape == depth.shape
    assert out.dtype == np.float32
    # Output mean ≈ input mean (filter preserves DC).
    assert abs(float(out.mean()) - float(depth.mean())) < 0.5
