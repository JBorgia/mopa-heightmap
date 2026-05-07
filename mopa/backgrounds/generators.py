"""Procedural background generators (numpy-only, no extra deps).

Public surface:

    generate_pattern(name, width, height, *, scale, angle, seed) -> np.ndarray
    guilloche_pattern(...)
    stripes_pattern(...)
    dots_pattern(...)
    halftone_pattern(...)
    checkers_pattern(...)

Returns a ``(H, W)`` float32 array in ``[0, 1]``. Higher value = brighter
output (matches the sculptok-bright_raised convention so the patterns
read as "raised" relief when they end up in the heightmap).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


PATTERN_NAMES: Tuple[str, ...] = (
    "guilloche",
    "stripes",
    "dots",
    "halftone",
    "checkers",
    "solid_black",
    "solid_white",
    "solid_grey",
)


def _rotation_meshgrid(
    width: int,
    height: int,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (xr, yr) coordinate arrays rotated by ``angle_deg``.

    Origin is the image centre so the rotation pivots on the middle
    of the canvas.
    """
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx -= width / 2.0
    yy -= height / 2.0
    theta = np.deg2rad(angle_deg)
    cs = np.cos(theta)
    sn = np.sin(theta)
    xr = xx * cs + yy * sn
    yr = -xx * sn + yy * cs
    return xr, yr


# ----------------------------------------------------------- guilloché

def guilloche_pattern(
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    angle: float = 0.0,
    seed: int = 0,
    n_curves: int = 6,
) -> np.ndarray:
    """Engine-turned guilloché. Composition of polar-coordinate sin curves.

    Looks like the engraved swirls on the back of a pocket watch or
    coin. ``scale`` stretches/compresses the rosette; higher
    ``n_curves`` adds frequencies for a denser pattern.
    """
    rng = np.random.default_rng(int(seed))
    xr, yr = _rotation_meshgrid(width, height, angle)
    # Polar coordinates, scaled.
    longest = max(width, height)
    radial_unit = max(longest * 0.5 / max(scale, 0.05), 1e-6)
    r = np.sqrt(xr * xr + yr * yr) / radial_unit
    theta = np.arctan2(yr, xr)

    # Sum of N harmonic sin waves at random frequencies + offsets.
    accum = np.zeros_like(r, dtype=np.float32)
    for _ in range(int(max(1, n_curves))):
        freq_r = float(rng.uniform(8.0, 22.0))
        freq_t = float(rng.integers(3, 12))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        accum += np.sin(freq_r * r + freq_t * theta + phase).astype(np.float32)
    # Normalise to [0, 1]. Sum-of-sines lives in [-N, N]; map via cosine
    # squash for a smoother look than pure linear scaling.
    norm = (np.sin(accum * 0.6) + 1.0) * 0.5
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------- stripes

def stripes_pattern(
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    angle: float = 0.0,
    seed: int = 0,
    duty: float = 0.5,
) -> np.ndarray:
    """Hard-edged stripe pattern at the given angle.

    ``scale`` = stripes per "200 px"; ``duty`` is the on-fraction of
    the stripe period (0.5 = symmetric).
    """
    del seed  # deterministic; seed kept for signature symmetry
    xr, _ = _rotation_meshgrid(width, height, angle)
    period_px = max(8.0, 200.0 / max(scale, 0.05))
    phase = (xr / period_px) % 1.0
    on = phase < float(np.clip(duty, 0.05, 0.95))
    return on.astype(np.float32)


# ----------------------------------------------------------- dots (Bridson-ish)

def dots_pattern(
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    angle: float = 0.0,
    seed: int = 0,
    coverage: float = 0.04,
) -> np.ndarray:
    """Soft circular dots placed on a jittered hex grid.

    Cheaper than full Bridson Poisson-disk sampling but visually similar
    for engraving stipple. ``scale`` controls dot density.
    """
    del angle  # dots are rotationally symmetric
    rng = np.random.default_rng(int(seed))
    spacing = max(6.0, 24.0 / max(scale, 0.05))
    radius = spacing * 0.35
    out = np.zeros((height, width), dtype=np.float32)

    # Hex grid centres with jitter.
    rows = int(height / (spacing * 0.866)) + 2
    cols = int(width / spacing) + 2
    for row in range(rows):
        for col in range(cols):
            cy = row * spacing * 0.866
            cx = col * spacing + (spacing * 0.5 if row % 2 else 0.0)
            cy += float(rng.uniform(-spacing * 0.15, spacing * 0.15))
            cx += float(rng.uniform(-spacing * 0.15, spacing * 0.15))
            if cx < -radius or cx >= width + radius: continue
            if cy < -radius or cy >= height + radius: continue
            x0 = max(0, int(cx - radius * 1.5))
            x1 = min(width, int(cx + radius * 1.5) + 1)
            y0 = max(0, int(cy - radius * 1.5))
            y1 = min(height, int(cy + radius * 1.5) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            # Soft dot via cosine falloff.
            dot = np.clip(1.0 - d / radius, 0.0, 1.0)
            np.maximum(out[y0:y1, x0:x1], dot, out=out[y0:y1, x0:x1])

    if coverage > 0.0:
        # Optional uniform dimming so the average coverage matches.
        out *= float(np.clip(coverage * 25.0, 0.1, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------- halftone

def halftone_pattern(
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    angle: float = 0.0,
    seed: int = 0,
    cell_value: float = 0.6,
) -> np.ndarray:
    """Constant-density halftone — circular dots on a regular grid at
    the given angle. Each dot is sized to match ``cell_value`` (0=empty
    cell, 1=full cell).
    """
    del seed
    cell_px = max(6.0, 16.0 / max(scale, 0.05))
    xr, yr = _rotation_meshgrid(width, height, angle)
    cx_in_cell = (xr / cell_px) % 1.0 - 0.5
    cy_in_cell = (yr / cell_px) % 1.0 - 0.5
    d = np.sqrt(cx_in_cell * cx_in_cell + cy_in_cell * cy_in_cell)
    # Solve dot radius from desired cell value (area fraction).
    target = float(np.clip(cell_value, 0.0, 1.0))
    dot_r = np.sqrt(target / np.pi) * 0.95  # hard cap below the cell edge
    inside = d < dot_r
    return inside.astype(np.float32)


# ----------------------------------------------------------- checkers

def checkers_pattern(
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    angle: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Two-cell checker. Useful for QA / registration patterns."""
    del seed
    xr, yr = _rotation_meshgrid(width, height, angle)
    cell_px = max(6.0, 32.0 / max(scale, 0.05))
    cx = np.floor(xr / cell_px).astype(np.int32)
    cy = np.floor(yr / cell_px).astype(np.int32)
    on = ((cx + cy) % 2 == 0)
    return on.astype(np.float32)


# ----------------------------------------------------------- solid fills

def solid_black_pattern(
    width: int, height: int, *, scale: float = 1.0, angle: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Flat black fill. Used to scrub the background out of the photo
    before sculptok sees it — sculptok focuses on the subject silhouette
    when the surroundings carry no signal."""
    del scale, angle, seed
    return np.zeros((height, width), dtype=np.float32)


def solid_white_pattern(
    width: int, height: int, *, scale: float = 1.0, angle: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Flat white fill. The mirror-image case — engraves the background
    flat to surface and lets the subject relief stand out as recessed."""
    del scale, angle, seed
    return np.ones((height, width), dtype=np.float32)


def solid_grey_pattern(
    width: int, height: int, *, scale: float = 1.0, angle: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """Mid-grey (0.5) fill. A neutral midpoint — useful when the photo's
    own background mean already sits near the centre and you just want
    to flatten variation without forcing extreme depth at the seam."""
    del scale, angle, seed
    return np.full((height, width), 0.5, dtype=np.float32)


# ----------------------------------------------------------- dispatch

_PATTERN_DISPATCH = {
    "guilloche": guilloche_pattern,
    "stripes": stripes_pattern,
    "dots": dots_pattern,
    "halftone": halftone_pattern,
    "checkers": checkers_pattern,
    "solid_black": solid_black_pattern,
    "solid_white": solid_white_pattern,
    "solid_grey": solid_grey_pattern,
}


def generate_pattern(
    name: str,
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    angle: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Dispatch to the named pattern. Raises ``KeyError`` for unknown names."""
    fn = _PATTERN_DISPATCH.get(name.lower())
    if fn is None:
        raise KeyError(f"Unknown pattern: {name!r}; available: {PATTERN_NAMES}")
    return fn(width, height, scale=scale, angle=angle, seed=seed)
