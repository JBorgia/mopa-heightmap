"""Accept a precomputed heightmap (sculptok / meshy / hand-authored) as
the depth source instead of running our own depth network.

The new product framing for this pipeline: the user brings a relief
heightmap from a higher-quality external tool (sculptok.com, meshy.ai's
image-to-3D + ortho-render, or a hand-painted bas-relief), plus the
original photo, and we deliver the *engraving toolchain* on top —
multi-pass planning, calibration, .lbrn2/.clb export, burn-time, QA,
color passes, signature. Sculptok provides the relief; we provide
everything else.

Design:

    photo + sculptok.png  →  load PNG (any bit depth, any polarity)
                          →  resize to match the photo
                          →  normalise polarity (bright=raised → LightBurn
                             "1.0=surface, 0.0=deepest" inside the subject)
                          →  apply BiRefNet subject mask (outside = 1.0
                             so the engraver leaves the background flat)
                          →  auto-stretch the in-subject range to span
                             [deep_limit, surface_limit] for the best
                             possible LightBurn 3D-Sliced dynamic range
                          →  hand back a depth array compatible with
                             ``infer_depth`` so service.render() doesn't
                             notice the swap

Polarity vocabulary (sculptok/meshy use display polarity, LightBurn uses
engraving polarity):

    bright_raised  — sculptok/meshy convention: bright pixels = raised
                     subject, dark pixels = deeper or background.
    dark_raised    — opposite convention: dark pixels = raised. Rare,
                     but legacy tools occasionally produce this.
    auto           — sample the four corners; if they're dark, assume
                     bright_raised (subject brighter than bg). If light,
                     assume dark_raised. The most common case is
                     bright_raised so that's the default fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "load_external_heightmap",
    "normalise_polarity",
    "auto_stretch_subject",
    "fit_external_heightmap_to_photo",
    "Polarity",
    "DEFAULT_POLARITY",
    "DEFAULT_AUTO_STRETCH",
    "DEFAULT_USE_SUBJECT_MASK",
    "DEFAULT_RESAMPLE",
    "EXTERNAL_DEPTH_DEEP_LIMIT",
    "EXTERNAL_DEPTH_SURFACE_LIMIT",
]


Polarity = Literal["bright_raised", "dark_raised", "auto"]


DEFAULT_POLARITY: Polarity = "bright_raised"
DEFAULT_AUTO_STRETCH: bool = True
DEFAULT_USE_SUBJECT_MASK: bool = True
DEFAULT_RESAMPLE: str = "realesrgan-x4plus"

# Engraving budget the auto-stretch fills inside the subject silhouette.
# These are conservative defaults compatible with most material profiles —
# the actual ``deep_limit`` / ``surface_limit`` from the loaded profile
# will further shape the curve in ``apply_tone_curve``.
EXTERNAL_DEPTH_DEEP_LIMIT: float = 0.02
EXTERNAL_DEPTH_SURFACE_LIMIT: float = 0.98


@dataclass(frozen=True)
class _LoadedHeightmap:
    """Internal struct passed between load → normalise → stretch."""

    array: np.ndarray         # float32 in [0, 1]
    source_path: Path
    source_size: Tuple[int, int]


def _read_grayscale_to_float(path: Path) -> np.ndarray:
    """Load a PNG/JPG/TIFF as float32 ``[0, 1]`` grayscale.

    Auto-detects 8 / 16 / 32-bit pixel formats.
    """
    img = Image.open(path)
    mode = img.mode
    if mode == "I;16" or mode == "I;16B" or mode == "I;16L":
        arr = np.asarray(img, dtype=np.uint16).astype(np.float32) / 65535.0
    elif mode == "I" or mode == "I;32":
        arr = np.asarray(img, dtype=np.int32).astype(np.float32)
        # 32-bit grayscale: map by max so we don't assume the range.
        arr = arr / max(float(arr.max()), 1.0)
    elif mode == "F":
        arr = np.asarray(img, dtype=np.float32)
    elif mode in ("L", "LA"):
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    else:
        # Color: convert to luminance via Rec.709 weights for stability.
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _detect_polarity(arr: np.ndarray, *, sample_size: int = 8) -> Polarity:
    """Sniff the four corners — if they're dark the subject is brighter."""
    h, w = arr.shape
    s = max(2, min(sample_size, h // 4, w // 4))
    corners = np.concatenate([
        arr[:s, :s].ravel(),
        arr[:s, -s:].ravel(),
        arr[-s:, :s].ravel(),
        arr[-s:, -s:].ravel(),
    ])
    return "bright_raised" if float(np.mean(corners)) < 0.4 else "dark_raised"


def normalise_polarity(arr: np.ndarray, polarity: Polarity) -> np.ndarray:
    """Convert any input polarity to "bright = raised, dark = recessed".

    Internally this stays the same as the input ``bright_raised``
    convention; we only flip when the input is ``dark_raised``. Auto
    detects via :func:`_detect_polarity`.
    """
    if polarity == "auto":
        polarity = _detect_polarity(arr)
    if polarity == "dark_raised":
        arr = 1.0 - arr
    elif polarity != "bright_raised":
        raise ValueError(f"Unknown polarity: {polarity!r}")
    return arr.astype(np.float32, copy=False)


def auto_stretch_subject(
    heightmap: np.ndarray,
    subject_alpha: Optional[np.ndarray],
    *,
    deep_limit: float = EXTERNAL_DEPTH_DEEP_LIMIT,
    surface_limit: float = EXTERNAL_DEPTH_SURFACE_LIMIT,
    background_value: float = 1.0,
) -> np.ndarray:
    """Stretch the in-subject heightmap to fill the engraving budget.

    The supplied ``heightmap`` arrives in "bright = raised" polarity.
    For LightBurn 3D Sliced (``black_is_deep=True`` convention) we want:
      * outside the subject → ``background_value`` (default 1.0 = no engraving),
      * inside the subject → percentile-stretched into ``[deep_limit, surface_limit]``
        (raised = closer to surface_limit, recessed = closer to deep_limit).

    When no subject alpha is provided, the whole frame is treated as
    subject (the operator is responsible for background polarity).
    """
    arr = heightmap.astype(np.float32, copy=False)
    if subject_alpha is None:
        subj_mask = np.ones_like(arr, dtype=bool)
    else:
        if subject_alpha.shape != arr.shape:
            raise ValueError(
                f"subject_alpha shape {subject_alpha.shape} does not match "
                f"heightmap shape {arr.shape}"
            )
        subj_mask = subject_alpha >= 0.5

    out = np.full_like(arr, float(background_value), dtype=np.float32)
    if subj_mask.any():
        inside = arr[subj_mask]
        p2, p98 = float(np.percentile(inside, 2.0)), float(np.percentile(inside, 98.0))
        span = max(p98 - p2, 1e-6)
        stretched = np.clip((inside - p2) / span, 0.0, 1.0)
        # bright_raised (1=surface, 0=deepest) → engraving range
        # [deep_limit, surface_limit]
        mapped = stretched * (surface_limit - deep_limit) + deep_limit
        out[subj_mask] = mapped.astype(np.float32)
    return out


def _resample_to(
    image: Image.Image,
    target_size: Tuple[int, int],
    resampler_key: str,
    device: str,
) -> Image.Image:
    """Upscale (or downscale) ``image`` to ``target_size`` using the
    super-resolution registry. Falls back to lanczos when the requested
    resolver isn't available."""
    target_w, target_h = target_size
    if image.size == target_size:
        return image
    src_long = max(image.size)
    tgt_long = max(target_size)
    # Only spin up the heavy SR network when actually upscaling. Going
    # down or sideways is plain PIL-LANCZOS territory.
    if tgt_long > src_long * 1.05:
        try:
            from .super_resolution import auto_upscale

            up = auto_upscale(
                image, target_long_side=tgt_long,
                resolver_key=resampler_key, device=device,
            )
            if up.size != target_size:
                up = up.resize(target_size, Image.LANCZOS)
            return up
        except Exception:
            return image.resize(target_size, Image.LANCZOS)
    return image.resize(target_size, Image.LANCZOS)


def load_external_heightmap(
    path: str | Path,
    *,
    target_size: Tuple[int, int] | None = None,
    polarity: Polarity = DEFAULT_POLARITY,
    resampler_key: str = DEFAULT_RESAMPLE,
    device: str = "cpu",
) -> np.ndarray:
    """Load + polarity-normalise an external heightmap.

    Returns a float32 ``(H, W)`` array in ``[0, 1]`` with the **bright =
    raised** convention. Subject masking and auto-stretch are deliberately
    NOT done here — they're separate so callers can plug in the BiRefNet
    alpha or use a pre-supplied mask.

    Parameters
    ----------
    path
        Filesystem path to the heightmap PNG/JPG/TIFF.
    target_size
        ``(W, H)`` to resize the loaded heightmap to. If omitted, the
        native size is preserved.
    polarity
        ``"bright_raised"`` (default), ``"dark_raised"``, or ``"auto"``.
    resampler_key
        Key into the super-resolution registry, used only when target
        size requests an upscale by more than 5 %. Default
        ``realesrgan-x4plus`` (BSD-3, fits in 4 GB VRAM).
    device
        Device hint for SR backends; ignored by lanczos / bicubic.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"External heightmap not found: {src}")

    arr = _read_grayscale_to_float(src)
    arr = normalise_polarity(arr, polarity)

    if target_size is not None and (arr.shape[1], arr.shape[0]) != target_size:
        as_image = Image.fromarray(
            (arr * 255.0 + 0.5).astype(np.uint8), mode="L",
        ).convert("RGB")
        as_image = _resample_to(as_image, target_size, resampler_key, device)
        # Real-ESRGAN's output is RGB; we want a single-channel float
        # heightmap. Drop to luminance via Rec.709 weights to keep the
        # bright_raised relationship stable.
        rgb = np.asarray(as_image.convert("RGB"), dtype=np.float32) / 255.0
        arr = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        ).astype(np.float32)
    return arr


def fit_external_heightmap_to_photo(
    heightmap_path: str | Path,
    photo_size: Tuple[int, int],
    *,
    subject_alpha: Optional[np.ndarray] = None,
    polarity: Polarity = DEFAULT_POLARITY,
    auto_stretch: bool = DEFAULT_AUTO_STRETCH,
    deep_limit: float = EXTERNAL_DEPTH_DEEP_LIMIT,
    surface_limit: float = EXTERNAL_DEPTH_SURFACE_LIMIT,
    background_value: float = 1.0,
    resampler_key: str = DEFAULT_RESAMPLE,
    device: str = "cpu",
) -> np.ndarray:
    """One-shot: load + resize + polarity-normalise + (optionally) stretch.

    The output is a final LightBurn-ready heightmap (``black_is_deep=True``
    convention: 1.0 = surface, 0.0 = deepest engraving), sized to match
    the source photo, with the background flattened to ``background_value``
    when a subject alpha is supplied.

    This is the canonical entry point service.render() calls when
    ``external_heightmap_path`` is set.
    """
    arr = load_external_heightmap(
        heightmap_path,
        target_size=photo_size,
        polarity=polarity,
        resampler_key=resampler_key,
        device=device,
    )
    if auto_stretch:
        return auto_stretch_subject(
            arr,
            subject_alpha,
            deep_limit=deep_limit,
            surface_limit=surface_limit,
            background_value=background_value,
        )
    # No stretch: just flatten background if mask available, else return
    # the bright_raised heightmap unchanged (caller maps it themselves).
    if subject_alpha is not None:
        out = np.where(subject_alpha >= 0.5, arr, np.float32(background_value))
        return out.astype(np.float32)
    return arr.astype(np.float32)
