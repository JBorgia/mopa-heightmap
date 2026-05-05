"""Derive per-pass masks from a finished heightmap.

The pass planner ([`stages.py`]) accepts a ``masks`` mapping keyed by
``PASS_KIND_*``; if a key is missing the pass falls back to an all-ones
mask, which makes Cleanup / Detail / Shading / Polish indistinguishable.
This module produces the per-pass masks the planner is asking for so the
exported ``.lbrn2`` actually represents the engraving recipe described
in IMPLEMENTATION_PLAN.md §4.

Pipeline (heightmap is float32 in ``[0, 1]`` with ``1.0 = surface``,
``0.0 = deepest engraving`` under ``black_is_deep=True``):

    Form     = subject silhouette (everywhere the heightmap is below 1.0)
    Cleanup  = edge-dilated ring around the Form silhouette
    Detail   = high-frequency band of the heightmap (small features)
    Shading  = mid-frequency band (cheekbones, fabric folds)
    Polish   = full subject mask (final dithered surface)
    Signature = empty by default — caller paints in the corner sigil

Each output is float32 in ``[0, 1]`` with the same shape as the input.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


__all__ = [
    "derive_pass_masks",
    "form_mask",
    "cleanup_mask",
    "detail_mask",
    "shading_mask",
    "polish_mask",
    "photo_tonal_mask",
    "DEFAULT_FORM_THRESHOLD",
    "DEFAULT_CLEANUP_RADIUS_PX",
    "DEFAULT_DETAIL_SIGMA_PX",
    "DEFAULT_SHADING_SIGMA_PX",
    "DEFAULT_PHOTO_TONAL_LEVELS",
    "DEFAULT_PHOTO_TONAL_STRENGTH",
]


# Heightmap value above which a pixel is considered "background" (no
# engraving). Anything strictly below this threshold belongs to the form.
DEFAULT_FORM_THRESHOLD: float = 0.985

# Edge ring radius for the cleanup pass — wide enough that small chatter
# along the silhouette gets a dedicated low-power pass without hitting the
# main subject body.
DEFAULT_CLEANUP_RADIUS_PX: int = 6

# Gaussian σ (in pixels) defining the cut-off between high-frequency
# detail (faces, embroidery) and mid-frequency shading (cheekbones, fabric
# folds). A 4-px sigma keeps eyes / lips on the Detail layer; a 24-px
# sigma sends macro form to the Shading layer.
DEFAULT_DETAIL_SIGMA_PX: float = 4.0
DEFAULT_SHADING_SIGMA_PX: float = 24.0

# Floyd-Steinberg dither levels for the photo-tonal pass. 16 gives a
# pleasant ordered-appearance dither; 256 is effectively continuous-tone
# (use when the engraver itself does the half-toning at the laser
# parameters layer).
DEFAULT_PHOTO_TONAL_LEVELS: int = 32

# Default tonal strength: the photo's luminance is multiplied by this
# before going through the dither + subject mask. 1.0 = full photo
# contrast; 0.5 cuts visual impact in half so the relief still reads.
DEFAULT_PHOTO_TONAL_STRENGTH: float = 0.7


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.5)
    try:
        import cv2

        ksize = max(3, int(round(sigma * 6)) | 1)
        return cv2.GaussianBlur(arr.astype(np.float32), (ksize, ksize),
                                sigmaX=sigma, sigmaY=sigma)
    except Exception:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr.astype(np.float32), sigma=sigma)


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    try:
        import cv2

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0
    except Exception:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask.astype(bool), iterations=radius)


def _erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    try:
        import cv2

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.erode(mask.astype(np.uint8), kernel) > 0
    except Exception:
        from scipy.ndimage import binary_erosion

        return binary_erosion(mask.astype(bool), iterations=radius)


def form_mask(
    heightmap: np.ndarray,
    *,
    threshold: float = DEFAULT_FORM_THRESHOLD,
    feather_px: int = 2,
) -> np.ndarray:
    """Subject silhouette — 1 wherever the heightmap is below ``threshold``."""
    binary = (heightmap < float(threshold)).astype(np.float32)
    if feather_px > 0:
        binary = _gaussian_blur(binary, feather_px)
    return np.clip(binary, 0.0, 1.0).astype(np.float32)


def cleanup_mask(
    heightmap: np.ndarray,
    *,
    threshold: float = DEFAULT_FORM_THRESHOLD,
    radius_px: int = DEFAULT_CLEANUP_RADIUS_PX,
) -> np.ndarray:
    """Edge ring around the form silhouette — dilation minus erosion.

    Used as a low-power, narrow pass that knocks down burr / re-cast slag
    that the Form pass leaves along the perimeter.
    """
    binary = heightmap < float(threshold)
    if not binary.any():
        return np.zeros_like(heightmap, dtype=np.float32)
    dilated = _dilate_binary(binary, radius_px)
    eroded = _erode_binary(binary, max(1, radius_px // 2))
    ring = dilated & ~eroded
    return _gaussian_blur(ring.astype(np.float32), max(1.0, radius_px / 3.0))


def detail_mask(
    heightmap: np.ndarray,
    *,
    sigma_px: float = DEFAULT_DETAIL_SIGMA_PX,
) -> np.ndarray:
    """High-frequency magnitude band — small features that need a tight raster.

    Computed as ``|heightmap - blur(heightmap, sigma)|`` and renormalised.
    A subject pixel with no high-frequency content (a flat cheek) lands
    near zero; an eye edge lands near one. Background is suppressed via
    multiplication with the form mask so we don't engrave detail that
    isn't there.
    """
    h = heightmap.astype(np.float32, copy=False)
    blurred = _gaussian_blur(h, sigma_px)
    band = np.abs(h - blurred)
    peak = float(band.max())
    if peak > 1e-6:
        band = band / peak
    return (band * form_mask(h)).astype(np.float32)


def shading_mask(
    heightmap: np.ndarray,
    *,
    sigma_px: float = DEFAULT_SHADING_SIGMA_PX,
    detail_sigma_px: float = DEFAULT_DETAIL_SIGMA_PX,
) -> np.ndarray:
    """Mid-frequency band: smooth depth structure between Detail and Form.

    Computed as the difference of two Gaussians: ``blur(σ_detail) − blur(σ_shade)``.
    This is the band that carries gentle gradients (cheekbone, fabric folds)
    and benefits from a soft, dithered pass with low power.
    """
    h = heightmap.astype(np.float32, copy=False)
    fine = _gaussian_blur(h, detail_sigma_px)
    coarse = _gaussian_blur(h, sigma_px)
    band = fine - coarse
    span = float(np.abs(band).max())
    if span > 1e-6:
        band = band / span
    return (np.clip(band * 0.5 + 0.5, 0.0, 1.0) * form_mask(h)).astype(np.float32)


def polish_mask(
    heightmap: np.ndarray,
    *,
    threshold: float = DEFAULT_FORM_THRESHOLD,
) -> np.ndarray:
    """Full subject mask — the polish pass touches every engraved pixel."""
    return form_mask(heightmap, threshold=threshold, feather_px=1)


def photo_tonal_mask(
    photo_rgb: np.ndarray,
    subject_alpha: np.ndarray | None,
    *,
    invert: bool = False,
    dither: bool = True,
    dither_levels: int = DEFAULT_PHOTO_TONAL_LEVELS,
    strength: float = DEFAULT_PHOTO_TONAL_STRENGTH,
) -> np.ndarray:
    """Convert the photo to a "fire-the-laser-more" intensity mask.

    Returns a float32 ``[0, 1]`` array where **higher value = more laser
    firing**. The pass-stack wiring (``_emit_pass_stack``) then converts
    that to a layer PNG via ``layer = 1.0 - mask * photo_tonal_depth``
    so dark photo regions land at the configured engraving depth and
    bright photo regions stay near the surface.

    Default polarity (``invert=False``) for engraving-on-bright-metal:

        * **Dark photo region** (beard, hair, eye sockets)
          → mask high → laser fires → metal carved darker.
        * **Bright photo region** (skin, lit fabric)
          → mask low → laser barely fires → metal stays as-is.
        * **Background** (outside the BiRefNet subject)
          → mask zero → laser doesn't fire.

    Use ``invert=True`` for unusual materials where the engraving
    process *adds* lightness (titanium anneal colors on dark steel).

    Parameters
    ----------
    photo_rgb
        ``(H, W, 3)`` uint8 RGB array of the source photo.
    subject_alpha
        Optional ``(H, W)`` float32 mask in ``[0, 1]`` from BiRefNet /
        rembg. The output is multiplied by this so we never engrave
        photo data onto the flat background. Pass ``None`` to skip
        gating.
    invert
        Flip the polarity above (bright photo fires laser, dark stays).
    dither
        Run Floyd–Steinberg dithering on the result. Default ``True``
        because most laser pulse-power curves are non-linear and
        ordered dither produces flatter midtones than continuous-tone
        gradients.
    dither_levels
        Quantisation steps when ``dither=True``. Lower → more visible
        pattern; higher → smoother.
    strength
        Multiplier on the photo's contribution before subject-gating.
        ``1.0`` is full contrast; lower values cut visual impact so the
        underlying carved relief still reads.
    """
    arr = np.asarray(photo_rgb)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.max(initial=0.0) > 1.5:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        luma = (
            0.2126 * arr[..., 0]
            + 0.7152 * arr[..., 1]
            + 0.0722 * arr[..., 2]
        )
    elif arr.ndim == 2:
        luma = arr
    else:
        luma = arr[..., 0]
    luma = np.clip(luma, 0.0, 1.0)
    # Default: engrave-more where the photo is dark. Caller flips for
    # contrarian materials.
    if not invert:
        luma = 1.0 - luma
    luma = luma * float(np.clip(strength, 0.0, 1.0))

    if subject_alpha is not None:
        if subject_alpha.shape != luma.shape:
            raise ValueError(
                f"subject_alpha shape {subject_alpha.shape} does not match "
                f"photo shape {luma.shape}"
            )
        luma = luma * subject_alpha.astype(np.float32, copy=False)

    if dither:
        from .heightmap import floyd_steinberg_dither

        luma = floyd_steinberg_dither(luma, levels=int(max(2, dither_levels)))

    return np.clip(luma, 0.0, 1.0).astype(np.float32)


def derive_pass_masks(
    heightmap: np.ndarray,
    *,
    form_threshold: float = DEFAULT_FORM_THRESHOLD,
    cleanup_radius_px: int = DEFAULT_CLEANUP_RADIUS_PX,
    detail_sigma_px: float = DEFAULT_DETAIL_SIGMA_PX,
    shading_sigma_px: float = DEFAULT_SHADING_SIGMA_PX,
) -> Dict[str, np.ndarray]:
    """Compute the four canonical raster pass masks in one shot.

    Returns a mapping suitable for ``stages.plan_passes(..., masks=...)``.
    The pre_clean and signature passes are intentionally absent — they
    use full-frame defaults or caller-supplied vector inputs.
    """
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must be 2-D; got shape {heightmap.shape}")

    from .stages import (
        PASS_KIND_CLEANUP,
        PASS_KIND_DETAIL,
        PASS_KIND_FORM,
        PASS_KIND_POLISH,
        PASS_KIND_SHADING,
    )

    return {
        PASS_KIND_FORM: form_mask(heightmap, threshold=form_threshold),
        PASS_KIND_CLEANUP: cleanup_mask(
            heightmap, threshold=form_threshold, radius_px=cleanup_radius_px,
        ),
        PASS_KIND_DETAIL: detail_mask(heightmap, sigma_px=detail_sigma_px),
        PASS_KIND_SHADING: shading_mask(
            heightmap, sigma_px=shading_sigma_px, detail_sigma_px=detail_sigma_px,
        ),
        PASS_KIND_POLISH: polish_mask(heightmap, threshold=form_threshold),
    }
