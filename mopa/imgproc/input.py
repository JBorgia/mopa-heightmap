"""Stage A image-processing primitives — input conditioning before depth inference.

Each function is independently togglable. They take a PIL Image (RGB) and
return a new PIL Image (RGB). No state, no globals.

Heavy / optional dependencies (rembg, real-esrgan, SAM) live in sibling
modules so this file imports nothing the core install doesn't already have.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class InputConditioningSettings:
    """Toggleable Stage A pipeline. Defaults are conservative no-ops where possible."""
    auto_orient: bool = True              # EXIF transpose
    white_balance: bool = False           # Gray-world AWB
    clahe: bool = False                   # CLAHE on L channel
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise: bool = False                 # Non-local-means
    denoise_strength: float = 5.0         # h parameter
    remove_specular: bool = False         # Threshold + inpaint
    specular_threshold: int = 245         # 0-255 brightness above which is "specular"
    specular_radius: int = 5              # Inpaint radius in px
    max_input_dim: int = 0                # 0 = no cap; otherwise downscale longest side


# ---------------------------------------------------------------- helpers
def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _as_bgr(image: Image.Image) -> np.ndarray:
    arr = np.asarray(_to_rgb(image))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _from_bgr(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- operations
def auto_orient(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag and drop it. Always safe."""
    return ImageOps.exif_transpose(_to_rgb(image))


def gray_world_white_balance(image: Image.Image) -> Image.Image:
    """Classic gray-world: scale each channel so its mean equals the global mean."""
    arr = np.asarray(_to_rgb(image), dtype=np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    if (means <= 1e-6).any():
        return image
    scaled = arr * (target / means)
    return Image.fromarray(np.clip(scaled, 0.0, 255.0).astype(np.uint8))


def clahe_lightness(image: Image.Image, clip_limit: float = 2.0, tile_grid: int = 8) -> Image.Image:
    """Equalize the L channel of LAB with CLAHE; preserves color."""
    bgr = _as_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    l_eq = clahe.apply(l)
    out = cv2.merge((l_eq, a, b))
    return _from_bgr(cv2.cvtColor(out, cv2.COLOR_LAB2BGR))


def denoise_nlm(image: Image.Image, strength: float = 5.0) -> Image.Image:
    """Non-local-means color denoise. `strength` maps to OpenCV's `h`."""
    bgr = _as_bgr(image)
    h = max(float(strength), 0.1)
    out = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
    )
    return _from_bgr(out)


def remove_specular_highlights(
    image: Image.Image,
    threshold: int = 245,
    inpaint_radius: int = 5,
) -> Image.Image:
    """Mask near-saturated pixels and inpaint them. Helps polished metal photos."""
    bgr = _as_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    if mask.sum() == 0:
        return image
    out = cv2.inpaint(bgr, mask, int(max(inpaint_radius, 1)), cv2.INPAINT_TELEA)
    return _from_bgr(out)


def cap_longest_side(image: Image.Image, max_dim: int) -> Image.Image:
    if max_dim <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


# ---------------------------------------------------------------- pipeline
def condition_input(
    image: Image.Image,
    settings: InputConditioningSettings | Mapping[str, object] | None = None,
) -> Image.Image:
    """Run the full Stage A pipeline in order. Each step is a no-op when disabled."""
    if settings is None:
        cfg = InputConditioningSettings()
    elif isinstance(settings, InputConditioningSettings):
        cfg = settings
    else:
        cfg = settings_from_mapping(settings)

    out = _to_rgb(image)
    if cfg.auto_orient:
        out = auto_orient(out)
    if cfg.max_input_dim:
        out = cap_longest_side(out, cfg.max_input_dim)
    if cfg.white_balance:
        out = gray_world_white_balance(out)
    if cfg.remove_specular:
        out = remove_specular_highlights(out, cfg.specular_threshold, cfg.specular_radius)
    if cfg.clahe:
        out = clahe_lightness(out, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.denoise:
        out = denoise_nlm(out, cfg.denoise_strength)
    return out


def settings_from_mapping(payload: Mapping[str, object]) -> InputConditioningSettings:
    cfg = InputConditioningSettings()
    for key in cfg.__dataclass_fields__:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, payload[key])
    return cfg
