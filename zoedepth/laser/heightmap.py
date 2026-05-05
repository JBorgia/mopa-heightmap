from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image


def _as_float32_2d(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D depth array, received shape {data.shape!r}")
    return data


def normalize_depth(
    depth: np.ndarray,
    near_percentile: float = 5.0,
    far_percentile: float = 95.0,
) -> np.ndarray:
    depth_map = _as_float32_2d(depth)
    finite_mask = np.isfinite(depth_map)
    if not finite_mask.any():
        raise ValueError("Depth map does not contain any finite values")

    finite_values = depth_map[finite_mask]
    near_value = float(np.percentile(finite_values, near_percentile))
    far_value = float(np.percentile(finite_values, far_percentile))
    if far_value <= near_value:
        far_value = near_value + 1e-6

    clipped = np.clip(depth_map, near_value, far_value)
    normalized = (clipped - near_value) / (far_value - near_value)
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    return normalized.astype(np.float32)


def orient_for_lightburn(normalized_depth: np.ndarray, black_is_deep: bool = True) -> np.ndarray:
    depth_map = _as_float32_2d(normalized_depth)
    if black_is_deep:
        return (1.0 - depth_map).astype(np.float32)
    return depth_map.astype(np.float32)


def apply_tone_curve(
    heightmap: np.ndarray,
    gamma: float = 1.0,
    contrast: float = 1.0,
    midtone_boost: float = 0.0,
    deep_limit: float = 0.0,
    surface_limit: float = 1.0,
) -> np.ndarray:
    mapped = np.clip(_as_float32_2d(heightmap), 0.0, 1.0)

    if gamma <= 0:
        raise ValueError("gamma must be greater than zero")
    mapped = np.power(mapped, gamma, dtype=np.float32)

    if contrast != 1.0:
        mapped = np.clip((mapped - 0.5) * contrast + 0.5, 0.0, 1.0)

    if midtone_boost:
        mapped = np.clip(mapped + (np.sin(np.pi * mapped) * midtone_boost), 0.0, 1.0)

    deep_limit = float(np.clip(deep_limit, 0.0, 1.0))
    surface_limit = float(np.clip(surface_limit, 0.0, 1.0))
    if surface_limit < deep_limit:
        raise ValueError("surface_limit must be greater than or equal to deep_limit")

    mapped = deep_limit + mapped * (surface_limit - deep_limit)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def flatten_background(
    heightmap: np.ndarray,
    threshold: float = 0.88,
    value: float = 1.0,
) -> np.ndarray:
    mapped = _as_float32_2d(heightmap).copy()
    mapped[mapped >= threshold] = np.clip(value, 0.0, 1.0)
    return mapped.astype(np.float32)


def smooth_heightmap(
    heightmap: np.ndarray,
    method: str = "bilateral",
    diameter: int = 9,
    strength: float = 0.08,
) -> np.ndarray:
    mapped = _as_float32_2d(heightmap)
    mode = (method or "none").lower()
    if mode in {"none", "off"}:
        return mapped.astype(np.float32)
    if mode == "bilateral":
        sigma_color = max(strength, 1e-6)
        sigma_space = max(float(diameter), 1.0)
        return cv2.bilateralFilter(mapped, int(max(diameter, 1)), sigma_color, sigma_space).astype(np.float32)
    if mode == "gaussian":
        sigma = max(strength, 1e-6)
        return cv2.GaussianBlur(mapped, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
    raise ValueError(f"Unsupported smoothing method: {method}")


def sharpen_heightmap(heightmap: np.ndarray, amount: float = 0.0, sigma: float = 2.0) -> np.ndarray:
    mapped = _as_float32_2d(heightmap)
    if amount <= 0:
        return mapped.astype(np.float32)

    blurred = cv2.GaussianBlur(mapped, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = np.clip(mapped + amount * (mapped - blurred), 0.0, 1.0)
    return sharpened.astype(np.float32)


def bilateral_cross_filter_depth(
    depth: np.ndarray,
    guide_rgb: np.ndarray | None,
    diameter: int = 9,
    sigma_color: float = 0.05,
    sigma_space: float = 8.0,
) -> np.ndarray:
    """Edge-preserving smoothing of a *depth* field guided by an RGB photo.

    Differs from :func:`joint_bilateral_refine` in two ways:
      * Operates on the raw depth (any scale, not clipped to ``[0, 1]``).
      * Returns a depth field with the same scale as the input.

    Used as a pre-normalisation stage that aligns depth edges to photo
    edges (hair / fabric silhouettes) without compressing dynamic range.
    Falls back to the input unchanged when ``cv2.ximgproc`` is missing.
    """
    if guide_rgb is None:
        return depth.astype(np.float32, copy=False)
    # ``ximgproc`` is in opencv-contrib only; check both module and the
    # specific function, since the regular ``opencv-python`` ships an
    # empty ximgproc namespace that has no ``jointBilateralFilter``.
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "jointBilateralFilter"):
        return depth.astype(np.float32, copy=False)

    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2-D; got shape {arr.shape}")

    guide = np.asarray(guide_rgb)
    if guide.ndim == 3 and guide.shape[2] == 3:
        guide_bgr = cv2.cvtColor(guide.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        guide_bgr = guide.astype(np.uint8)
    if guide_bgr.shape[:2] != arr.shape:
        guide_bgr = cv2.resize(
            guide_bgr, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_AREA,
        )
    guide_f32 = guide_bgr.astype(np.float32) / 255.0

    # Normalise depth into [0, 1] for the bilateral pass, then map back so
    # the absolute depth scale is preserved.
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    span = max(hi - lo, 1e-6)
    norm = (arr - lo) / span
    refined = ximgproc.jointBilateralFilter(
        guide_f32, norm.astype(np.float32),
        int(max(diameter, 1)), float(sigma_color), float(sigma_space),
    )
    return (refined * span + lo).astype(np.float32)


def joint_bilateral_refine(
    heightmap: np.ndarray,
    guide_rgb: np.ndarray | None,
    diameter: int = 9,
    sigma_color: float = 0.1,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """Edge-aware refinement: smooth the heightmap while respecting RGB edges.

    Falls back to the input unchanged when no guide is given or when
    `cv2.ximgproc` isn't installed.
    """
    mapped = _as_float32_2d(heightmap)
    if guide_rgb is None:
        return mapped

    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "jointBilateralFilter"):
        return mapped

    guide = np.asarray(guide_rgb)
    if guide.ndim == 3 and guide.shape[2] == 3:
        guide_bgr = cv2.cvtColor(guide.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        guide_bgr = guide.astype(np.uint8)

    if guide_bgr.shape[:2] != mapped.shape:
        guide_bgr = cv2.resize(
            guide_bgr,
            (mapped.shape[1], mapped.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # jointBilateralFilter requires src and joint to share depth (CV_8U or CV_32F).
    guide_f32 = (guide_bgr.astype(np.float32) / 255.0)
    refined = ximgproc.jointBilateralFilter(
        guide_f32,
        mapped,
        int(max(diameter, 1)),
        float(sigma_color),
        float(sigma_space),
    )
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def floyd_steinberg_dither(heightmap: np.ndarray, levels: int = 256) -> np.ndarray:
    """Quantize a [0,1] heightmap to `levels` gray steps using Floyd-Steinberg.

    Returns a float32 array still in [0,1]. Caller chooses bit depth on save.
    Useful when collapsing a 16-bit master to 8-bit without banding.
    """
    src = _as_float32_2d(heightmap).copy()
    levels = max(int(levels), 2)
    step = 1.0 / (levels - 1)
    h, w = src.shape
    out = np.zeros_like(src)
    for y in range(h):
        for x in range(w):
            old = src[y, x]
            new = round(old / step) * step
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                src[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    src[y + 1, x - 1] += err * (3 / 16)
                src[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    src[y + 1, x + 1] += err * (1 / 16)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def posterize_for_passes(heightmap: np.ndarray, n_passes: int) -> np.ndarray:
    """Quantize a [0,1] heightmap into `n_passes` discrete depth steps.

    Useful as a preview of what the engraver will physically deliver in 3D
    Sliced mode: each pass adds one step of depth, so the achievable
    resolution is `n_passes + 1` distinct heights (including no-burn).

    Returns a float32 array still in [0,1] for downstream rendering.
    """
    src = _as_float32_2d(heightmap)
    n_passes = max(int(n_passes), 1)
    if n_passes == 1:
        return np.where(src >= 0.5, 1.0, 0.0).astype(np.float32)
    levels = n_passes + 1
    quantized = np.round(src * (levels - 1)) / (levels - 1)
    return np.clip(quantized, 0.0, 1.0).astype(np.float32)


def process_depth_to_heightmap(
    depth: np.ndarray,
    settings: Mapping[str, object],
    *,
    guide_rgb: np.ndarray | None = None,
) -> np.ndarray:
    normalized = normalize_depth(
        depth,
        near_percentile=float(settings.get("near_percentile", 5.0)),
        far_percentile=float(settings.get("far_percentile", 95.0)),
    )
    mapped = orient_for_lightburn(
        normalized,
        black_is_deep=bool(settings.get("black_is_deep", True)),
    )

    # Stage B — photo-detail injection. Done early (before tone curve) so the
    # tone curve can still rescale the combined signal into the engraving
    # range. No-op unless detail_mode != "off" and detail_strength > 0.
    if guide_rgb is not None:
        from .detail import apply_detail_injection, settings_from_mapping as _detail_from
        mapped = apply_detail_injection(mapped, guide_rgb, _detail_from(settings))

    mapped = apply_tone_curve(
        mapped,
        gamma=float(settings.get("gamma", 1.0)),
        contrast=float(settings.get("contrast", 1.0)),
        midtone_boost=float(settings.get("midtone_boost", 0.0)),
        deep_limit=float(settings.get("deep_limit", 0.0)),
        surface_limit=float(settings.get("surface_limit", 1.0)),
    )

    if bool(settings.get("flatten_background", False)):
        mapped = flatten_background(
            mapped,
            threshold=float(settings.get("background_threshold", 0.88)),
            value=float(settings.get("background_value", 1.0)),
        )

    mapped = smooth_heightmap(
        mapped,
        method=str(settings.get("smooth", "bilateral")),
        diameter=int(settings.get("smooth_diameter", 9)),
        strength=float(settings.get("smooth_strength", 0.08)),
    )

    if bool(settings.get("edge_refine", False)) and guide_rgb is not None:
        mapped = joint_bilateral_refine(
            mapped,
            guide_rgb,
            diameter=int(settings.get("edge_refine_diameter", 9)),
            sigma_color=float(settings.get("edge_refine_sigma_color", 0.08)),
            sigma_space=float(settings.get("edge_refine_sigma_space", 6.0)),
        )

    mapped = sharpen_heightmap(
        mapped,
        amount=float(settings.get("sharpen", 0.0)),
        sigma=float(settings.get("sharpen_sigma", 2.0)),
    )

    lut_payload = settings.get("calibration_lut")
    if lut_payload:
        # Imported lazily to keep heightmap.py free of cross-module imports
        # for callers that never use calibration.
        from .lut import CalibrationLUT

        try:
            lut = CalibrationLUT.from_dict(lut_payload) if isinstance(lut_payload, Mapping) else lut_payload
            mapped = lut.apply(
                mapped,
                target_depth_um=settings.get("target_depth_um"),
            )
        except (ValueError, TypeError):
            # Bad LUT data shouldn't break the export; fall through silently.
            pass

    posterize_passes = int(settings.get("posterize_passes", 0) or 0)
    if posterize_passes > 0:
        mapped = posterize_for_passes(mapped, posterize_passes)

    if bool(settings.get("dither", False)):
        mapped = floyd_steinberg_dither(
            mapped,
            levels=int(settings.get("dither_levels", 256)),
        )

    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def to_uint8(heightmap: np.ndarray) -> np.ndarray:
    mapped = np.clip(_as_float32_2d(heightmap), 0.0, 1.0)
    return np.round(mapped * 255.0).astype(np.uint8)


def to_uint16(heightmap: np.ndarray) -> np.ndarray:
    mapped = np.clip(_as_float32_2d(heightmap), 0.0, 1.0)
    return np.round(mapped * 65535.0).astype(np.uint16)


def save_heightmap_uint8(heightmap: np.ndarray, output_path: str | Path) -> Path:
    target = Path(output_path)
    Image.fromarray(to_uint8(heightmap), mode="L").save(target)
    return target


def save_heightmap_uint16(heightmap: np.ndarray, output_path: str | Path) -> Path:
    target = Path(output_path)
    Image.fromarray(to_uint16(heightmap), mode="I;16").save(target)
    return target