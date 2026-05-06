"""Refinement-pass mask helpers.

Sculptok produces the depth layer (the .lbrn2's 3DSliced bitmap). The
refinement layers add separate physical features on top of the carved
relief — color zones, photo-tonal shading, signature text — each with
its own LightBurn cut setting. Their masks come from the *photo* (or
from explicit user input), not from the heightmap.

This module is just the photo-tonal mask helper for now. Photo-detail
(hair / fur / eye edges via Sobel/Laplacian on the photo), eye-line,
frame, and cut-outline masks land here as those features ship.
"""
from __future__ import annotations

import numpy as np


__all__ = [
    "photo_tonal_mask",
    "DEFAULT_PHOTO_TONAL_LEVELS",
    "DEFAULT_PHOTO_TONAL_STRENGTH",
]


# Floyd-Steinberg dither levels for the photo-tonal pass. 16 gives a
# pleasant ordered-appearance dither; 256 is effectively continuous-tone
# (use when the engraver itself does the half-toning at the laser
# parameters layer).
DEFAULT_PHOTO_TONAL_LEVELS: int = 32

# Default tonal strength: the photo's luminance is multiplied by this
# before going through the dither + subject mask. 1.0 = full photo
# contrast; 0.5 cuts visual impact in half so the relief still reads.
DEFAULT_PHOTO_TONAL_STRENGTH: float = 0.7


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
