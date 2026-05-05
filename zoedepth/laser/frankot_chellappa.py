"""Frankot–Chellappa surface integration.

Reconstruct a height field ``z(x, y)`` from a dense gradient field
``(p, q) = (∂z/∂x, ∂z/∂y)`` by solving the Poisson equation in the Fourier
domain. The classical solution from Frankot & Chellappa (1988):

.. math::

    Z(u, v) = \\frac{-j\\,u\\,P(u, v) - j\\,v\\,Q(u, v)}{u^2 + v^2}

with the DC bin set to zero (the integrator is gauge-invariant — the mean
height is undetermined). All math is done in float64 inside, the result is
returned as float32 to match the rest of the heightmap pipeline.

This module is intentionally pure-NumPy — no torch, no SciPy — so it can run
inside the live preview loop without warming up CUDA. ~50 ms for 1024² on a
typical desktop CPU.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


__all__ = [
    "integrate_normals",
    "integrate_gradients",
    "normals_to_gradients",
    "DEFAULT_PAD_MODE",
    "DEFAULT_PAD_FRACTION",
    "DC_BIN_VALUE",
    "EPS_NZ",
]


# ---------------------------------------------------------------- constants

# How we extend the field beyond its native bounds before FFT. ``"reflect"``
# (mirror without repeating the edge sample) gives the cleanest cancellation
# of the artificial step the Frankot solver would otherwise see at the
# wrap-around seam.
DEFAULT_PAD_MODE: str = "reflect"

# Pad width as a fraction of the longer source dimension. 0.25 means we
# reconstruct on a domain 1.5× larger than the source, then crop back.
# Empirically suppresses edge ringing without inflating FFT cost too much.
DEFAULT_PAD_FRACTION: float = 0.25

# The DC (zero-frequency) Fourier coefficient is undetermined for a Poisson
# problem; setting it to zero fixes the integration constant (mean height).
DC_BIN_VALUE: complex = 0.0 + 0.0j

# Floor for ``|nz|`` when converting normals to gradients, to avoid 1/0
# blow-up at silhouette edges where the surface normal is nearly tangent
# to the image plane.
EPS_NZ: float = 1e-6


# ---------------------------------------------------------------- normals -> gradients

def normals_to_gradients(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert unit normals ``N = (Nx, Ny, Nz)`` into surface gradients.

    Standard image-space convention (``z`` points *out of* the image, ``y``
    points down): ``p = ∂z/∂x = -Nx / Nz`` and ``q = ∂z/∂y = -Ny / Nz``.

    Parameters
    ----------
    normals
        Array of shape ``(H, W, 3)`` whose last axis is the unit normal.

    Returns
    -------
    p, q
        Two ``(H, W)`` float32 arrays. Pixels where ``|Nz| < EPS_NZ`` get
        zero gradient (treated as occluding contour, no constraint).
    """
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(
            f"normals must have shape (H, W, 3); got {normals.shape}"
        )
    nx = normals[..., 0].astype(np.float64)
    ny = normals[..., 1].astype(np.float64)
    nz = normals[..., 2].astype(np.float64)
    safe = np.where(np.abs(nz) < EPS_NZ, np.sign(nz) * EPS_NZ + EPS_NZ, nz)
    p = -nx / safe
    q = -ny / safe
    invalid = np.abs(nz) < EPS_NZ
    p[invalid] = 0.0
    q[invalid] = 0.0
    return p.astype(np.float32), q.astype(np.float32)


# ---------------------------------------------------------------- core integrator

def _frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pure FFT solver. Operates on whatever shape it receives (no padding)."""
    h, w = p.shape
    # Frequency grids in cycles / sample. ``fftfreq`` already returns the
    # signed bins in the FFT-native order, so no fftshift is needed.
    fx = np.fft.fftfreq(w).reshape(1, w)
    fy = np.fft.fftfreq(h).reshape(h, 1)
    P = np.fft.fft2(p.astype(np.float64))
    Q = np.fft.fft2(q.astype(np.float64))
    # Use 2π·fx because Frankot's derivation assumes angular frequency.
    u = 2.0 * np.pi * fx
    v = 2.0 * np.pi * fy
    denom = u * u + v * v
    numer = (-1j * u) * P + (-1j * v) * Q
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(denom > 0.0, numer / denom, DC_BIN_VALUE)
    z = np.fft.ifft2(Z).real
    return z.astype(np.float32)


def integrate_gradients(
    p: np.ndarray,
    q: np.ndarray,
    *,
    pad_fraction: float = DEFAULT_PAD_FRACTION,
    pad_mode: str = DEFAULT_PAD_MODE,
) -> np.ndarray:
    """Integrate a gradient field into a height field.

    Pads the input by ``pad_fraction`` of the longer side (mirror by default)
    to suppress the FFT's implicit periodic-boundary ringing, integrates,
    and crops back to the source resolution.
    """
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must share shape; got {p.shape} vs {q.shape}"
        )
    if p.ndim != 2:
        raise ValueError(f"p, q must be 2-D; got shape {p.shape}")
    if not 0.0 <= pad_fraction < 1.0:
        raise ValueError(
            f"pad_fraction must be in [0, 1); got {pad_fraction}"
        )

    h, w = p.shape
    pad = int(round(max(h, w) * float(pad_fraction)))
    if pad > 0:
        p_pad = np.pad(p, pad, mode=pad_mode)
        q_pad = np.pad(q, pad, mode=pad_mode)
        z_pad = _frankot_chellappa(p_pad, q_pad)
        z = z_pad[pad : pad + h, pad : pad + w]
    else:
        z = _frankot_chellappa(p, q)
    return np.ascontiguousarray(z, dtype=np.float32)


def integrate_normals(
    normals: np.ndarray,
    *,
    pad_fraction: float = DEFAULT_PAD_FRACTION,
    pad_mode: str = DEFAULT_PAD_MODE,
) -> np.ndarray:
    """Convenience wrapper: ``(H, W, 3)`` normals → ``(H, W)`` height field.

    The output is centered at zero (mean = 0); downstream code is responsible
    for any rescale into the laser's [0, 1] heightmap range.
    """
    p, q = normals_to_gradients(normals)
    z = integrate_gradients(p, q, pad_fraction=pad_fraction, pad_mode=pad_mode)
    return (z - float(z.mean())).astype(np.float32)
