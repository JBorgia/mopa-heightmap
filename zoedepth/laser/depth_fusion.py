"""Multi-resolution depth fusion (PromptDA-style).

Run a depth backend at several input resolutions and combine them so the
*global* plane comes from the lowest-resolution estimate (which sees the
whole subject and gets the silhouette / large surfaces right) and the
*high-frequency edges* come from the highest-resolution estimate (which
sees pore-level detail but often loses the global shape to local biases).

The fusion is a Laplacian-pyramid blend implemented with separable
Gaussian low-pass — pure NumPy, no SciPy required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "fuse_depths",
    "MultiResolutionDepth",
    "DEFAULT_FUSION_SCALES",
    "DEFAULT_LOWPASS_SIGMA_PX",
    "DEFAULT_FUSION_HIGHPASS_WEIGHT",
    "MIN_FUSION_SCALES",
]


# ----------------------------------------------------------- constants

# Default resolutions (longest side) at which to run the depth backend.
# 512 catches the global plane, 1024 is the sweet spot, 2048 only adds
# value when the source photo is high-quality.
DEFAULT_FUSION_SCALES: Tuple[int, ...] = (512, 1024, 2048)

# Gaussian sigma (pixels in the *output* frame) that defines the cut-off
# between "low-frequency global plane" and "high-frequency detail". 24 px
# corresponds to ~3% of a 768 px image — about the size of a forehead.
DEFAULT_LOWPASS_SIGMA_PX: float = 24.0

# How much of the high-frequency residual from finer scales to add back on
# top of the low-frequency base. 1.0 = full residual (sharpest, possibly
# noisy); 0.0 = ignore detail.
DEFAULT_FUSION_HIGHPASS_WEIGHT: float = 1.0

# At least 1 scale (degenerate to single-resolution); 2+ is where fusion
# actually does anything useful.
MIN_FUSION_SCALES: int = 1

# Numerical guard for divisions when normalising fused output.
_EPS_FUSION: float = 1e-8


# ----------------------------------------------------------- helpers

def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _separable_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2-D Gaussian blur (pure NumPy, edge-padded)."""
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)
    k = _gaussian_kernel_1d(sigma)
    pad = len(k) // 2
    padded = np.pad(arr.astype(np.float32, copy=False), pad, mode="edge")
    rows = np.zeros_like(padded)
    for offset, weight in enumerate(k):
        rows += weight * np.roll(padded, shift=offset - pad, axis=1)
    cols = np.zeros_like(padded)
    for offset, weight in enumerate(k):
        cols += weight * np.roll(rows, shift=offset - pad, axis=0)
    return cols[pad:-pad, pad:-pad].astype(np.float32, copy=False)


def _resize_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    pil = Image.fromarray(depth.astype(np.float32, copy=False), mode="F")
    return np.asarray(pil.resize((target_w, target_h), Image.BILINEAR), dtype=np.float32)


# ----------------------------------------------------------- fusion

def fuse_depths(
    depths: Sequence[np.ndarray],
    *,
    sigma_px: float = DEFAULT_LOWPASS_SIGMA_PX,
    highpass_weight: float = DEFAULT_FUSION_HIGHPASS_WEIGHT,
) -> np.ndarray:
    """Fuse multiple depth maps via a Laplacian-pyramid blend.

    All inputs are resampled to the resolution of the *largest* input
    (assumed to carry the most pixel-accurate edges). The fused output
    has the same shape as that largest input.

    The base low-frequency band is taken from the *smallest* depth's
    upsampled blur (cleanest global geometry); higher-resolution inputs
    contribute only their high-frequency residual.
    """
    if len(depths) < MIN_FUSION_SCALES:
        raise ValueError(
            f"fuse_depths requires at least {MIN_FUSION_SCALES} input(s); "
            f"got {len(depths)}"
        )
    arrays = [np.asarray(d, dtype=np.float32) for d in depths]
    for i, a in enumerate(arrays):
        if a.ndim != 2:
            raise ValueError(f"depth[{i}] must be 2-D, got shape {a.shape}")

    # Sort by area; smallest is the global-plane source, largest is the
    # canvas we render into.
    arrays_sorted = sorted(arrays, key=lambda a: a.size)
    smallest, *rest = arrays_sorted
    target = arrays_sorted[-1]
    h, w = target.shape

    base = _separable_gaussian(_resize_depth(smallest, h, w), sigma_px)
    fused = base.copy()
    if highpass_weight > 0.0:
        for a in rest:
            up = _resize_depth(a, h, w)
            highpass = up - _separable_gaussian(up, sigma_px)
            fused = fused + float(highpass_weight) * highpass
    return fused.astype(np.float32, copy=False)


# ----------------------------------------------------------- runner

@dataclass(frozen=True)
class MultiResolutionDepth:
    """Wrap a depth backend so it runs at several scales then fuses.

    The wrapped backend must expose ``infer_pil(image, **kwargs) ->
    np.ndarray`` (the same contract :class:`HeightmapService` uses).
    """

    backend: Any
    scales: Tuple[int, ...] = DEFAULT_FUSION_SCALES
    sigma_px: float = DEFAULT_LOWPASS_SIGMA_PX
    highpass_weight: float = DEFAULT_FUSION_HIGHPASS_WEIGHT

    def infer_pil(self, image: Image.Image, **kwargs: Any) -> np.ndarray:
        if not self.scales:
            raise ValueError("MultiResolutionDepth.scales must be non-empty")
        depths: list[np.ndarray] = []
        w, h = image.size
        long_side = max(w, h)
        for scale_px in self.scales:
            if scale_px <= 0:
                raise ValueError(f"scale {scale_px} must be positive")
            if scale_px == long_side:
                resized = image
            else:
                ratio = scale_px / float(long_side)
                resized = image.resize(
                    (max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))),
                    Image.BILINEAR,
                )
            depths.append(np.asarray(self.backend.infer_pil(resized, **kwargs), dtype=np.float32))
        return fuse_depths(depths, sigma_px=self.sigma_px, highpass_weight=self.highpass_weight)
