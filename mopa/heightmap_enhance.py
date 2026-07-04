"""Post-processing enhancements for photo-derived heightmaps.

Applied AFTER polarity normalisation and optional auto-stretch, BEFORE
polarity-invert.  Designed to push Sculptok / monocular-depth output
closer to coin-relief quality by sharpening depth-zone boundaries and
expanding the effective dynamic range.

Why this matters
----------------
Photo-to-depth AI (Sculptok, MiDaS, ZoeDepth, Depth-Anything) produces
*photographic* gradients — smooth, naturalistic, low-contrast in the
mid-range.  Professional coin heightmaps are orthographic Z-buffer
renders of 3D sculpts: very high contrast, sharp transitions between
depth zones (rim, field, portrait, background).

These transforms move the output in that direction without requiring a
3D model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["enhance_for_engraving", "ENHANCE_MODES"]

ENHANCE_MODES = ("off", "portrait", "standard", "coin")

# ── Internal helpers ──────────────────────────────────────────────────────────

def _percentile_stretch(
    arr: np.ndarray,
    mask: Optional[np.ndarray],
    lo_pct: float,
    hi_pct: float,
) -> np.ndarray:
    region = arr[mask] if mask is not None else arr.ravel()
    lo = float(np.percentile(region, lo_pct))
    hi = float(np.percentile(region, hi_pct))
    span = max(hi - lo, 1e-6)
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def _gamma(arr: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law curve.  gamma > 1 pushes midtones deeper (more contrast)."""
    return np.power(np.clip(arr, 0.0, 1.0), gamma).astype(np.float32)


def _clahe(arr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation applied to depth values."""
    try:
        import cv2  # type: ignore[import-untyped]
        u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
        clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        return clahe_op.apply(u8).astype(np.float32) / 255.0
    except ImportError:
        return arr


def _unsharp_mask(arr: np.ndarray, sigma: float = 1.5, strength: float = 0.25) -> np.ndarray:
    """Sharpen depth-zone transitions.  strength controls how much edge detail is boosted."""
    from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
    blurred = gaussian_filter(arr, sigma=sigma).astype(np.float32)
    return np.clip(arr + strength * (arr - blurred), 0.0, 1.0).astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def enhance_for_engraving(
    heightmap: np.ndarray,
    mode: str,
    *,
    subject_alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply engraving-optimised post-processing to a normalised [0, 1] heightmap.

    Parameters
    ----------
    heightmap
        Float32 ``(H, W)`` in bright-raised convention ``[0, 1]``.
    mode
        ``'off'``       — passthrough; no modification (default).
        ``'portrait'``  — gentle contrast stretch + mild gamma (1.2).
                         Good for face portraits where you want a natural look.
        ``'standard'``  — moderate stretch + gamma (1.35) + light unsharp mask.
                         Good starting point for most subjects.
        ``'coin'``      — aggressive stretch + gamma (1.5) + CLAHE + unsharp mask.
                         Pushes depth zones toward the high-contrast look of
                         3D-rendered coin relief.
    subject_alpha
        Optional float32 alpha mask.  When provided, percentile stretch is
        computed over subject pixels only so the background value is not
        compressed into the subject range.
    """
    if mode not in ENHANCE_MODES or mode == "off":
        return heightmap.astype(np.float32, copy=False)

    arr = heightmap.astype(np.float32, copy=True)
    mask = (subject_alpha >= 0.5) if subject_alpha is not None else None

    if mode == "portrait":
        arr = _percentile_stretch(arr, mask, lo_pct=5.0, hi_pct=95.0)
        arr = _gamma(arr, 1.2)

    elif mode == "standard":
        arr = _percentile_stretch(arr, mask, lo_pct=3.0, hi_pct=97.0)
        arr = _gamma(arr, 1.35)
        arr = _unsharp_mask(arr, sigma=1.5, strength=0.2)

    elif mode == "coin":
        arr = _percentile_stretch(arr, mask, lo_pct=2.0, hi_pct=98.0)
        arr = _gamma(arr, 1.5)
        arr = _clahe(arr, clip_limit=2.5, tile=8)
        arr = _unsharp_mask(arr, sigma=1.5, strength=0.3)

    return np.clip(arr, 0.0, 1.0).astype(np.float32)
