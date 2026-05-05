"""Closed-loop calibration: photo of an engraved ramp → ``CalibrationLUT``.

The full workflow:

    1. Operator burns ``preview.create_calibration_ramp()`` on the actual
       material at the speed/power they plan to use for production work.
    2. They photograph or flatbed-scan the engraved sample under controlled
       lighting (a phone snap on a desk works; flatbed scan is best).
    3. They feed that photo into :func:`calibration_lut_from_ramp_photo`,
       supply the deepest-step measured depth in microns (one calliper
       reading or the laser's quoted spec for that material/power), and
       get back a :class:`zoedepth.laser.lut.CalibrationLUT` ready to drop
       into a profile YAML via
       :func:`zoedepth.laser.profiles.write_lut_to_profile`.

This module never invents physics; it just converts a controlled
photometric measurement into the empirical (gray → depth) sample list the
LUT already consumes. The fit is just ``np.interp`` linear interpolation
(monotonic-enforcing); we additionally surface fit-quality diagnostics
so the operator can spot uneven exposure / scratches that would skew the
calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .lut import CalibrationLUT


__all__ = [
    "measure_engraved_ramp",
    "calibration_lut_from_ramp_photo",
    "RampMeasurement",
    "DEFAULT_RAMP_GRAY_LEVELS",
    "DEFAULT_RAMP_AXIS",
]


# Canonical gray levels of the calibration ramp produced by
# ``preview.create_calibration_ramp()``. Ordered surface→deepest so a
# horizontal ramp reads left=surface (255) → right=deepest (0).
DEFAULT_RAMP_GRAY_LEVELS: Tuple[int, ...] = (
    255, 230, 204, 179, 153, 128, 102, 77, 51, 26, 0,
)

# Axis along which the ramp's bands are tiled. ``"horizontal"`` matches
# ``create_calibration_ramp`` defaults; flip to ``"vertical"`` for portrait
# orientation scans.
DEFAULT_RAMP_AXIS: str = "horizontal"


@dataclass(frozen=True)
class RampMeasurement:
    """Result of extracting per-step photometry from a ramp photo."""

    # Median grayscale (0..1 normalised) per band, in the band order the
    # ramp was burned in (surface → deepest).
    band_luminance: Tuple[float, ...]
    # Per-band depth in microns (after rescaling against ``max_depth_um``).
    band_depth_um: Tuple[float, ...]
    # 0 if every consecutive depth grew; otherwise the count of inversions
    # (bands shallower than the previous one). A non-zero value usually
    # means uneven lighting or a misaligned crop.
    monotonic_violations: int
    # Optional fit-quality scalar — lower = cleaner. Equals the std-dev
    # of (measured - linearly-interpolated) residuals over the band centres.
    residual_rms: float


def _crop_image(
    image: Image.Image,
    crop: Optional[Tuple[int, int, int, int]],
) -> Image.Image:
    if crop is None:
        return image
    left, top, right, bottom = crop
    if right <= left or bottom <= top:
        raise ValueError(f"crop ({crop}) is empty or inverted")
    return image.crop((int(left), int(top), int(right), int(bottom)))


def _slice_band_medians(
    arr: np.ndarray,
    n_steps: int,
    axis: str,
) -> np.ndarray:
    """Slice a 2-D grayscale array into ``n_steps`` equal bands and return
    the median pixel value of each band, normalised to ``[0, 1]``."""
    if axis == "horizontal":
        bands = np.array_split(arr, n_steps, axis=1)
    elif axis == "vertical":
        bands = np.array_split(arr, n_steps, axis=0)
    else:
        raise ValueError(f"axis must be 'horizontal' or 'vertical'; got {axis!r}")
    medians = np.array([np.median(b) for b in bands], dtype=np.float64)
    return medians / 255.0


def measure_engraved_ramp(
    photo: Image.Image,
    *,
    n_steps: int = len(DEFAULT_RAMP_GRAY_LEVELS),
    axis: str = DEFAULT_RAMP_AXIS,
    crop: Optional[Tuple[int, int, int, int]] = None,
    invert: bool = False,
    max_depth_um: Optional[float] = None,
) -> RampMeasurement:
    """Extract per-step optical density from a photo of an engraved ramp.

    Parameters
    ----------
    photo
        PIL image of the burned ramp. Color is ignored; we operate on
        luminance. A flatbed-scanned region with even illumination
        produces the cleanest fit, but a phone snap of the ramp under
        diffuse light works.
    n_steps
        How many bands the ramp had when it was burned. Defaults to 11
        (the canonical :func:`preview.create_calibration_ramp` output).
    axis
        ``"horizontal"`` (default; bands tile left→right) or ``"vertical"``.
    crop
        Optional ``(left, top, right, bottom)`` rectangle in pixel coords
        to limit the analysis to. If you photograph the engraved sample
        on a tabletop, crop the photo to just the ramp region first.
    invert
        Set to ``True`` when the engraving creates *highlights* (oxidation
        / titanium-anneal colors) on a darker base, so depth and luminance
        run in the same direction in the photo. Default ``False`` assumes
        deeper engraving = darker pixel.
    max_depth_um
        Optional. If provided, scales the normalised band depths into
        absolute microns (``max_depth_um`` is the deepest measured step).
        If omitted, the depth values are returned in ``[0, 1]`` relative
        units — useful for diagnostics, but ``CalibrationLUT.apply()``
        needs absolute units.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}")

    cropped = _crop_image(photo.convert("L"), crop)
    arr = np.asarray(cropped, dtype=np.float32)
    if arr.ndim != 2 or min(arr.shape) < n_steps:
        raise ValueError(
            f"cropped photo too small for {n_steps}-band analysis "
            f"(shape={arr.shape})"
        )

    band_lum = _slice_band_medians(arr, n_steps, axis)

    # Convert luminance → relative depth. Surface band ought to be the
    # brightest (no engraving), deepest band the darkest. When
    # ``invert=True`` the polarity is flipped (highlight materials).
    surface_lum = band_lum[0] if not invert else band_lum[-1]
    deepest_lum = band_lum[-1] if not invert else band_lum[0]
    span = abs(surface_lum - deepest_lum)
    if span < 1e-6:
        raise ValueError(
            "Surface and deepest bands have indistinguishable brightness; "
            "is the photo cropped to just the ramp?"
        )
    if invert:
        rel = (band_lum - deepest_lum) / span
    else:
        rel = (surface_lum - band_lum) / span
    rel = np.clip(rel, 0.0, 1.0)

    if max_depth_um is None:
        depth_um = rel
    else:
        if max_depth_um <= 0:
            raise ValueError(f"max_depth_um must be > 0; got {max_depth_um}")
        depth_um = rel * float(max_depth_um)

    # Diagnostics.
    diffs = np.diff(depth_um)
    monotonic_violations = int((diffs < -1e-6).sum())
    residual_rms = float(_residual_rms(depth_um))

    return RampMeasurement(
        band_luminance=tuple(float(x) for x in band_lum),
        band_depth_um=tuple(float(x) for x in depth_um),
        monotonic_violations=monotonic_violations,
        residual_rms=residual_rms,
    )


def _residual_rms(depths: np.ndarray) -> float:
    """RMS of (depth − linear-fit-of-depth) — proxy for noise / fit quality."""
    n = len(depths)
    if n < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, n)
    line = np.linspace(depths[0], depths[-1], n)
    return float(np.sqrt(np.mean((depths - line) ** 2)))


def calibration_lut_from_ramp_photo(
    photo: Image.Image,
    *,
    max_depth_um: float,
    n_steps: int = len(DEFAULT_RAMP_GRAY_LEVELS),
    gray_levels: Optional[Sequence[float]] = None,
    axis: str = DEFAULT_RAMP_AXIS,
    crop: Optional[Tuple[int, int, int, int]] = None,
    invert: bool = False,
    note: str = "",
) -> Tuple[CalibrationLUT, RampMeasurement]:
    """Build a :class:`CalibrationLUT` directly from a ramp photo.

    Returns the LUT plus the diagnostic measurement so callers can warn
    on monotonic violations / high residual.
    """
    measurement = measure_engraved_ramp(
        photo,
        n_steps=n_steps,
        axis=axis,
        crop=crop,
        invert=invert,
        max_depth_um=max_depth_um,
    )
    if gray_levels is None:
        # The visual ramp tiles bands as (surface, …, deepest) = (255, …, 0),
        # but ``CalibrationLUT.apply()`` consumes the LUT in the engraver's
        # frame where gray=0 means "no engraving" (depth=0) and gray=255 means
        # "full engraving" (depth=max). So we reverse the canonical visual
        # gray sequence here to match ``band_depth_um`` (surface→deepest =
        # ``[0, …, max_depth_um]``) and end up with samples that satisfy the
        # apply() math: gray_to_depth(0) → 0, gray_to_depth(255) → max.
        gray_levels = tuple(reversed(DEFAULT_RAMP_GRAY_LEVELS[:n_steps]))
    if len(gray_levels) != n_steps:
        raise ValueError(
            f"gray_levels length ({len(gray_levels)}) does not match n_steps "
            f"({n_steps})"
        )
    lut = CalibrationLUT.from_measurements(
        depths_um=measurement.band_depth_um,
        gray_levels=gray_levels,
        note=note or f"Auto-fit from ramp photo (max_depth_um={max_depth_um})",
    )
    return lut, measurement
