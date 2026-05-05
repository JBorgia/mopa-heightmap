"""Tests for :mod:`zoedepth.laser.depth_unsharp`."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.depth_unsharp import (
    DEFAULT_BLEND,
    DEFAULT_GAMMA,
    EPS_GRAD,
    gradient_domain_compress,
)


def test_constants_have_documented_values():
    assert DEFAULT_GAMMA == 0.7
    assert DEFAULT_BLEND == 0.5
    assert EPS_GRAD == 1e-6


def test_gamma_one_is_no_op():
    """gamma=1 leaves gradients unchanged; the output equals the input."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(32, 32)).astype(np.float32)
    out = gradient_domain_compress(z, gamma=1.0, blend=1.0)
    assert np.allclose(out, z)


def test_blend_zero_is_no_op():
    """blend=0 returns the depth unchanged regardless of gamma."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(32, 32)).astype(np.float32)
    out = gradient_domain_compress(z, gamma=0.5, blend=0.0)
    assert np.allclose(out, z)


def test_compresses_large_gradients_amplifies_small():
    """For gamma<1, a steep ramp should flatten and a low-contrast field
    should pick up amplitude near edges."""
    # Steep ramp: 0..10 over 32 cols. Compression should reduce the span.
    ramp = np.tile(np.linspace(0.0, 10.0, 32, dtype=np.float32), (32, 1))
    out = gradient_domain_compress(ramp, gamma=0.5, blend=1.0)
    # Span should shrink; mean stays close to input mean (FC re-anchored).
    assert (out.max() - out.min()) < (ramp.max() - ramp.min())
    assert abs(float(out.mean()) - float(ramp.mean())) < 1e-3


def test_preserves_shape_and_dtype():
    z = np.zeros((16, 24), dtype=np.float32)
    out = gradient_domain_compress(z, gamma=0.7, blend=0.5)
    assert out.shape == (16, 24)
    assert out.dtype == np.float32


def test_rejects_gamma_out_of_range():
    z = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="gamma"):
        gradient_domain_compress(z, gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        gradient_domain_compress(z, gamma=1.5)


def test_rejects_blend_out_of_range():
    z = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="blend"):
        gradient_domain_compress(z, blend=-0.1)
    with pytest.raises(ValueError, match="blend"):
        gradient_domain_compress(z, blend=1.5)


def test_rejects_non_2d_input():
    z = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        gradient_domain_compress(z)


def test_finite_output_on_constant_field():
    """Flat input has zero gradient — divisions must guard against 0/0."""
    z = np.full((16, 16), 0.5, dtype=np.float32)
    out = gradient_domain_compress(z, gamma=0.5, blend=1.0)
    assert np.all(np.isfinite(out))
    # Output mean is preserved (no DC drift).
    assert abs(float(out.mean()) - 0.5) < 1e-3


def test_blend_is_linear_interpolation():
    """blend must linearly interpolate between input and full compression."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=(16, 16)).astype(np.float32)
    full = gradient_domain_compress(z, gamma=0.6, blend=1.0)
    half = gradient_domain_compress(z, gamma=0.6, blend=0.5)
    expected = 0.5 * z + 0.5 * full
    assert np.allclose(half, expected, atol=1e-5)
