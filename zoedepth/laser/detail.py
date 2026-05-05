"""Stage B — photo-detail injection on top of a smooth depth heightmap.

ZoeDepth gives global form (silhouette + body shape) but cannot resolve
fine surface texture: facial features, fabric weave, jewelry, embroidery.
For laser bas-relief we want both. This module blends photo-derived signals
on top of the depth heightmap so the final engraving has rich surface detail.

Conventions (match the rest of the laser pipeline):
    - heightmap is float32 in [0, 1]
    - 1.0 = surface (no engraving), 0.0 = max depth
    - photo_rgb is uint8 HxWx3 (or float32 0..1)
    - black_is_deep=True is the project default; "luminance" mode therefore
      uses photo luminance directly so shadows push the heightmap down.

Modes:
    "off"       — pass-through.
    "luminance" — blend a*L into the heightmap. Captures shadows/highlights.
    "highpass"  — add a*(L - blur(L)) into the heightmap. Captures local
                  texture without shifting the overall form.
    "both"      — luminance with strength a, plus high-pass with strength a/2.

Subject mask (optional): use the heightmap itself as a soft mask so the
flat far-plane background isn't perturbed. Anywhere heightmap < threshold
is left alone; anywhere above threshold gets full detail injection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


_VALID_MODES = ("off", "luminance", "highpass", "both")


@dataclass
class DetailSettings:
    mode: str = "off"
    strength: float = 0.0          # 0 = no detail, 1 = mostly photo
    highpass_radius: int = 9       # gaussian sigma in pixels for high-pass
    subject_mask: bool = True      # mask injection by heightmap (skip background)
    invert: bool = False           # flip luminance polarity (white-engraves-deep materials)


def settings_from_mapping(payload: Mapping[str, object] | None) -> DetailSettings:
    out = DetailSettings()
    if not payload:
        return out
    mode = str(payload.get("detail_mode", out.mode)).lower()
    # Tolerate UI suffixes like "luminance (experimental)" — keep just the
    # first bare word so the dropdown is free to annotate options.
    head = mode.split("(", 1)[0].strip().split()
    if head:
        mode = head[0]
    if mode in _VALID_MODES:
        out.mode = mode
    out.strength = float(payload.get("detail_strength", out.strength))
    out.highpass_radius = int(payload.get("detail_highpass_radius", out.highpass_radius))
    out.subject_mask = bool(payload.get("detail_subject_mask", out.subject_mask))
    out.invert = bool(payload.get("detail_invert", out.invert))
    return out


def _luminance(photo_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(photo_rgb)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0
    if arr.ndim == 2:
        return np.clip(arr, 0.0, 1.0)
    if arr.shape[-1] >= 3:
        # Rec.709 luma.
        return np.clip(0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2], 0.0, 1.0)
    return np.clip(arr[..., 0], 0.0, 1.0)


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.5, float(sigma))
    try:
        import cv2
        ksize = max(3, int(2 * round(3.0 * sigma) + 1))
        return cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    except Exception:
        # Pure-numpy separable fallback (slower but always available).
        radius = max(1, int(round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        # Horizontal then vertical (1-D convolutions via np.apply_along_axis).
        tmp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, arr)
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, tmp)


def _resize_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    try:
        import cv2
        return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        return np.asarray(img.resize((target_w, target_h), Image.LANCZOS), dtype=np.float32) / 255.0


def _subject_mask(heightmap: np.ndarray, threshold: float = 0.05, margin: float = 0.10) -> np.ndarray:
    """Smooth 0..1 mask: 0 where heightmap is far/deep (background), 1 where raised.

    Uses a smoothstep around `threshold` of width `margin`.
    """
    lo = max(0.0, threshold - margin * 0.5)
    hi = min(1.0, threshold + margin * 0.5)
    if hi <= lo:
        hi = lo + 1e-3
    t = np.clip((heightmap - lo) / (hi - lo), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def apply_detail_injection(
    heightmap: np.ndarray,
    photo_rgb: np.ndarray | None,
    settings: DetailSettings,
) -> np.ndarray:
    """Blend photo-derived detail into a depth-derived heightmap.

    Returns a new float32 array clipped to [0, 1]. If mode is "off",
    photo is None, or strength is zero, returns the input unchanged.
    """
    if settings.mode == "off" or settings.strength <= 0.0 or photo_rgb is None:
        return heightmap.astype(np.float32, copy=False)

    h = heightmap.astype(np.float32, copy=False)
    H, W = h.shape[-2], h.shape[-1]

    L = _luminance(photo_rgb)
    L = _resize_to(L, H, W)
    if settings.invert:
        L = 1.0 - L

    alpha = float(np.clip(settings.strength, 0.0, 1.0))
    mask = _subject_mask(h) if settings.subject_mask else np.ones_like(h, dtype=np.float32)

    if settings.mode in ("luminance", "both"):
        # Centered around the heightmap's local mean so we ADD detail without
        # destroying the depth-derived form. Specifically, replace by a
        # weighted blend: out = (1-w)*h + w*L_aligned, where L_aligned is
        # luminance shifted to share the same mean as the masked region.
        masked = h * mask
        m_h = masked.sum() / max(mask.sum(), 1.0)
        m_L = (L * mask).sum() / max(mask.sum(), 1.0)
        L_aligned = np.clip(L + (m_h - m_L), 0.0, 1.0)
        w = alpha * mask
        h = (1.0 - w) * h + w * L_aligned

    if settings.mode in ("highpass", "both"):
        sigma = max(1.0, float(settings.highpass_radius) / 2.0)
        hp = L - _gaussian_blur(L, sigma)
        # Limit high-pass excursion so we don't blow out reserves.
        hp = np.clip(hp, -0.5, 0.5)
        # Halve the strength when used alongside luminance so the two
        # signals don't compound into mush.
        hp_alpha = alpha * (0.5 if settings.mode == "both" else 1.0)
        h = h + hp_alpha * hp * mask

    return np.clip(h, 0.0, 1.0).astype(np.float32)
