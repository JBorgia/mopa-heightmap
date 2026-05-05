"""Tests for :mod:`zoedepth.laser.frankot_chellappa`."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.frankot_chellappa import (
    DC_BIN_VALUE,
    DEFAULT_PAD_FRACTION,
    DEFAULT_PAD_MODE,
    EPS_NZ,
    integrate_gradients,
    integrate_normals,
    normals_to_gradients,
)


# ----------------------------------------------------------- pinned constants

def test_constants_have_documented_values():
    assert DEFAULT_PAD_MODE == "reflect"
    assert DEFAULT_PAD_FRACTION == 0.25
    assert DC_BIN_VALUE == 0.0 + 0.0j
    assert EPS_NZ == 1e-6


# ----------------------------------------------------------- normals -> gradients

def test_normals_to_gradients_planar_surface_is_zero():
    h, w = 8, 8
    flat = np.tile([0.0, 0.0, 1.0], (h, w, 1)).astype(np.float32)
    p, q = normals_to_gradients(flat)
    assert np.allclose(p, 0.0)
    assert np.allclose(q, 0.0)


def test_normals_to_gradients_tilted_surface_recovers_slope():
    # 45° tilt to the right: surface z = x; ∂z/∂x = 1; normal = (-1/√2, 0, 1/√2)
    n = np.tile([-1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)], (4, 4, 1)).astype(np.float32)
    p, q = normals_to_gradients(n)
    assert np.allclose(p, 1.0, atol=1e-5)
    assert np.allclose(q, 0.0, atol=1e-5)


def test_normals_to_gradients_silhouette_safe():
    """Pixels with Nz ≈ 0 must produce zero gradient (no 1/0 blow-up)."""
    n = np.tile([1.0, 0.0, 0.0], (4, 4, 1)).astype(np.float32)
    p, q = normals_to_gradients(n)
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(q))
    assert np.allclose(p, 0.0)
    assert np.allclose(q, 0.0)


def test_normals_to_gradients_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        normals_to_gradients(np.zeros((4, 4, 4), dtype=np.float32))


# ----------------------------------------------------------- integrator

def test_integrator_constant_gradient_returns_zero_field():
    # Constant gradients live entirely in the DC bin, which the Poisson
    # solver intentionally zeros (the integration constant is gauge-free).
    # Document the behaviour rather than pretending we recover the ramp.
    h, w = 16, 16
    p = np.full((h, w), 0.5, dtype=np.float32)
    q = np.zeros((h, w), dtype=np.float32)
    z = integrate_gradients(p, q, pad_fraction=0.0)
    assert z.shape == (h, w)
    assert np.allclose(z, 0.0, atol=1e-5)


def test_integrator_recovers_paraboloid():
    # z = 0.5 (x² + y²)  ⇒  p = x, q = y; integrate and check curvature.
    h, w = 64, 64
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32).reshape(1, w)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    # Convert math gradients to per-pixel gradients (∂z/∂i = ∂z/∂x · dx/di).
    dxdi = (2.0 / (w - 1))
    dydi = (2.0 / (h - 1))
    p = (xs * dxdi).astype(np.float32) * np.ones_like(ys)
    q = (ys * dydi).astype(np.float32) * np.ones_like(xs)
    z = integrate_gradients(p, q, pad_fraction=0.25)
    # The recovered surface should be paraboloid up to an additive constant.
    centred = z - z.mean()
    truth = 0.5 * (xs * xs + ys * ys)
    truth = truth - truth.mean()
    # Compare correlation rather than absolute scale (FFT integrator is
    # exact in slope, so shapes line up); allow loose rms tolerance for
    # padding-induced low-frequency leakage.
    err = np.sqrt(np.mean((centred - truth) ** 2)) / (truth.std() + 1e-9)
    assert err < 0.15


def test_integrator_rejects_shape_mismatch():
    p = np.zeros((4, 4), dtype=np.float32)
    q = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="share shape"):
        integrate_gradients(p, q)


def test_integrator_rejects_invalid_pad_fraction():
    p = q = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="pad_fraction"):
        integrate_gradients(p, q, pad_fraction=1.0)
    with pytest.raises(ValueError, match="pad_fraction"):
        integrate_gradients(p, q, pad_fraction=-0.1)


def test_integrate_normals_returns_zero_mean():
    # Synthetic hemisphere normals (analytic surface).
    h, w = 32, 32
    xs = np.linspace(-0.9, 0.9, w, dtype=np.float32).reshape(1, w)
    ys = np.linspace(-0.9, 0.9, h, dtype=np.float32).reshape(h, 1)
    r2 = xs * xs + ys * ys
    nz = np.sqrt(np.maximum(0.0, 1.0 - r2))
    nx = xs * np.ones_like(ys)
    ny = ys * np.ones_like(xs)
    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    z = integrate_normals(normals, pad_fraction=0.25)
    assert z.shape == (h, w)
    assert abs(float(z.mean())) < 1e-5
