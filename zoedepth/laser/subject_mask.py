"""Subject-isolation masks for the heightmap pipeline.

Stage that produces a soft alpha mask (float32 in ``[0, 1]``, 1 = subject)
which downstream stages use to:

* hard-clamp the background to a known plane (no engraving outside the
  subject silhouette),
* gate detail injection so background noise doesn't get etched,
* feed the per-pass color planner with a cutout silhouette.

Backend choice mirrors :mod:`zoedepth.laser.backends`:

* ``rembg`` — pure-CPU ONNX, MIT, ~170 MB weights. Default-on.
* ``birefnet`` — SOTA dichotomous segmentation, MIT, ~220 M params. Default-on
  when the user has the GPU for it; falls back to rembg silently if loading
  fails (so install never blocks a preview).
* ``rmbg2`` — same architecture as BiRefNet but CC-BY-NC-4.0; opt-in only.
* ``threshold`` — depth-percentile fallback that needs no weights at all.
  Used when the user is offline / has no model installed.

License policy (see ``IMPLEMENTATION_PLAN.md`` §11): default-on backends are
Apache-2.0 / MIT only. Non-commercial backends require
``InferenceSettings.allow_nc_weights`` to be enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "SubjectMaskerSpec",
    "register_masker",
    "get_masker",
    "list_maskers",
    "load_masker",
    "SubjectMaskResult",
    "compose_mask_with_heightmap",
    # tunable defaults — exported so callers can override without re-import.
    "DEFAULT_BINARY_THRESHOLD",
    "DEFAULT_FEATHER_PX",
    "DEFAULT_BACKGROUND_PLANE",
    "DEFAULT_THRESHOLD_BACKEND_PERCENTILE",
    "RGBA_ALPHA_MAX",
]


# --------------------------------------------------------------------- constants

# Cut-off used to convert the soft alpha into a binary subject region for
# hard background flattening. 0.5 = "more subject than background".
DEFAULT_BINARY_THRESHOLD: float = 0.5

# Feather radius (pixels) applied to the binary edge before alpha blending,
# so flattened background doesn't produce a stair-step on the laser. Matches
# IMPLEMENTATION_PLAN.md §Phase 2 default of 3 px.
DEFAULT_FEATHER_PX: int = 3

# Heightmap value to write into the background after flattening.
# 1.0 == "laser surface, no engraving" under the standard black_is_deep
# convention (see :func:`zoedepth.laser.heightmap.orient_for_lightburn`).
DEFAULT_BACKGROUND_PLANE: float = 1.0

# Percentile of the heightmap used by the no-weights ``threshold`` backend
# to guess where the subject ends. 90th percentile assumes the subject is
# the brighter (closer-to-surface) majority of the frame.
DEFAULT_THRESHOLD_BACKEND_PERCENTILE: float = 90.0

# 8-bit alpha range; ONNX seg models return uint8 alpha planes that we
# normalise to float32 in [0, 1].
RGBA_ALPHA_MAX: int = 255


# --------------------------------------------------------------------- registry

@dataclass(frozen=True)
class SubjectMaskerSpec:
    """Metadata + loader for one subject-mask backend."""

    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]      # device -> object with .infer(PIL) -> np.ndarray


_REGISTRY: Dict[str, SubjectMaskerSpec] = {}


def register_masker(spec: SubjectMaskerSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Subject masker already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_masker(key: str) -> SubjectMaskerSpec | None:
    return _REGISTRY.get(key)


def list_maskers(include_opt_in: bool = True) -> Tuple[SubjectMaskerSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_masker(key: str, device: str) -> Tuple[Any, str]:
    """Instantiate a masker by key, returning ``(instance, device)``."""
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No subject masker registered for: {key!r}")
    return spec.loader(device), device


# --------------------------------------------------------------------- result type

@dataclass(frozen=True)
class SubjectMaskResult:
    """Output of any masker: float32 alpha, plus diagnostics."""

    alpha: np.ndarray                    # H×W float32 in [0, 1]; 1 = subject
    backend: str                         # which backend produced it
    source_size: Tuple[int, int]         # (W, H) of the source image at infer time


# ------------------------------------------------------------------- backends

class _RembgMasker:
    """Lightweight wrapper around :mod:`rembg` returning float32 alpha.

    rembg's ``remove`` function returns an RGBA PIL image; we drop RGB and
    keep the alpha plane normalised to ``[0, 1]``. Choice of session model
    is delegated to the user via the ``rembg`` env / cache because the
    default ``u2net`` silhouette is good enough for general subjects and the
    portrait-tuned ``u2net_human_seg`` activates automatically when present.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name  # None = rembg default selection

    def infer(self, image: Image.Image) -> np.ndarray:
        from rembg import remove, new_session

        rgba = remove(
            image.convert("RGB"),
            session=new_session(self._model_name) if self._model_name else None,
        )
        if not isinstance(rgba, Image.Image):
            rgba = Image.fromarray(np.asarray(rgba))
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        alpha = np.asarray(rgba.split()[-1], dtype=np.float32) / float(RGBA_ALPHA_MAX)
        return alpha


class _BiRefNetMasker:
    """BiRefNet via the official HF ``ZhengPeng7/BiRefNet`` checkpoint.

    Returns a float32 alpha resampled to the source image size.
    """

    # BiRefNet's training resolution. Inference at this square size matches
    # the published metrics; we resample back to the source size after.
    INFER_SIDE: int = 1024
    # ImageNet normalisation — required by the BiRefNet weights.
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device

    def _preprocess(self, image: Image.Image) -> "Any":
        import torch

        rgb = image.convert("RGB").resize(
            (self.INFER_SIDE, self.INFER_SIDE), Image.BICUBIC
        )
        arr = np.asarray(rgb, dtype=np.float32) / 255.0  # 0..1
        arr = (arr - np.asarray(self.IMAGENET_MEAN, dtype=np.float32)) / \
              np.asarray(self.IMAGENET_STD, dtype=np.float32)
        # HWC -> NCHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        return tensor

    def infer(self, image: Image.Image) -> np.ndarray:
        import torch

        with torch.inference_mode():
            preds = self._model(self._preprocess(image))
        # BiRefNet returns a list of multi-stage predictions; take the last
        # (highest-resolution head) and run sigmoid for [0, 1] alpha.
        if isinstance(preds, (list, tuple)):
            logits = preds[-1]
        else:
            logits = preds
        alpha = torch.sigmoid(logits)[0, 0].detach().to("cpu").float().numpy()
        # Resample to source image size (PIL uses W, H ordering).
        target_w, target_h = image.size
        alpha_pil = Image.fromarray((alpha * RGBA_ALPHA_MAX).astype(np.uint8))
        alpha_pil = alpha_pil.resize((target_w, target_h), Image.BICUBIC)
        return np.asarray(alpha_pil, dtype=np.float32) / float(RGBA_ALPHA_MAX)


class _ThresholdMasker:
    """Zero-dependency fallback: derive a mask from the depth/heightmap alone.

    Picks pixels whose value is in the upper percentile of the input. Works
    when the subject is the closest object in the frame (the common case for
    product/portrait engraving). Useful for offline mode and CI.
    """

    def __init__(self, percentile: float = DEFAULT_THRESHOLD_BACKEND_PERCENTILE) -> None:
        self._percentile = float(percentile)

    def infer(self, image: Image.Image) -> np.ndarray:
        # Convert to luminance, treat brightest pixels as subject. The
        # heightmap-based caller path passes the heightmap as a grayscale
        # PIL image so this works uniformly.
        gray = np.asarray(image.convert("L"), dtype=np.float32) / float(RGBA_ALPHA_MAX)
        cutoff = float(np.percentile(gray, self._percentile))
        # Soft transition over a small range above cutoff for nicer edges.
        return np.clip((gray - cutoff) / max(1e-3, 1.0 - cutoff), 0.0, 1.0)


def _make_rembg_loader(model_name: str | None) -> Callable[[str], Any]:
    def _loader(device: str) -> _RembgMasker:
        del device  # rembg picks ONNX provider itself
        return _RembgMasker(model_name=model_name)
    return _loader


def _make_birefnet_loader(repo_id: str) -> Callable[[str], Any]:
    def _loader(device: str) -> _BiRefNetMasker:
        # Lazy import — transformers/torch hub are heavy.
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            repo_id, trust_remote_code=True
        ).to(device).eval()
        return _BiRefNetMasker(model, device)
    return _loader


def _make_threshold_loader(percentile: float) -> Callable[[str], Any]:
    def _loader(device: str) -> _ThresholdMasker:
        del device
        return _ThresholdMasker(percentile=percentile)
    return _loader


# Default-on (Apache-2.0 / MIT).
register_masker(SubjectMaskerSpec(
    key="threshold",
    label="Depth threshold (no weights, offline)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,
    loader=_make_threshold_loader(DEFAULT_THRESHOLD_BACKEND_PERCENTILE),
))
register_masker(SubjectMaskerSpec(
    key="rembg",
    label="rembg (u2net, MIT)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,           # ONNX-CPU; GPU is optional
    loader=_make_rembg_loader(None),
))
register_masker(SubjectMaskerSpec(
    key="rembg_human",
    label="rembg portrait (u2net_human_seg, MIT)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,
    loader=_make_rembg_loader("u2net_human_seg"),
))
register_masker(SubjectMaskerSpec(
    key="birefnet",
    label="BiRefNet General (MIT, recommended for GPU)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=True,
    vram_estimate_mb=3500,
    loader=_make_birefnet_loader("ZhengPeng7/BiRefNet"),
))
# Opt-in only (CC-BY-NC-4.0).
register_masker(SubjectMaskerSpec(
    key="rmbg2",
    label="RMBG-2.0 (BRIA, CC-BY-NC-4.0)",
    license="CC-BY-NC-4.0",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=3500,
    loader=_make_birefnet_loader("briaai/RMBG-2.0"),
))


# ----------------------------------------------------------- composition helper

def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Disk-shaped binary dilation, used to feather the mask boundary."""
    if radius <= 0:
        return mask
    try:
        import cv2
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0
    except ImportError:
        # Pure-NumPy max-filter fallback (slower but always available).
        from numpy.lib.stride_tricks import sliding_window_view
        pad = radius
        padded = np.pad(mask.astype(np.uint8), pad, mode="edge")
        win = sliding_window_view(padded, (2 * pad + 1, 2 * pad + 1))
        return win.max(axis=(-2, -1)) > 0


def _gaussian_blur_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Light Gaussian blur on the alpha channel. Mirrors detail.py's idiom."""
    if radius <= 0:
        return alpha
    try:
        import cv2
        # cv2 expects an odd kernel size; sigma derived from radius matches
        # the convention used elsewhere in this codebase (sigma = radius/2).
        ksize = 2 * radius + 1
        sigma = max(0.5, radius / 2.0)
        return cv2.GaussianBlur(alpha, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    except ImportError:
        # Three-pass box blur ≈ Gaussian.
        from scipy.ndimage import uniform_filter
        out = alpha.astype(np.float32)
        for _ in range(3):
            out = uniform_filter(out, size=2 * radius + 1, mode="nearest")
        return out


def compose_mask_with_heightmap(
    heightmap: np.ndarray,
    alpha: np.ndarray,
    *,
    background_value: float = DEFAULT_BACKGROUND_PLANE,
    binary_threshold: float = DEFAULT_BINARY_THRESHOLD,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> np.ndarray:
    """Flatten the background of ``heightmap`` to ``background_value``.

    The mask is binarised at ``binary_threshold``, optionally dilated and
    Gaussian-blurred by ``feather_px`` for a soft edge, then alpha-blended:

    .. code-block::

        out = alpha_soft * heightmap + (1 - alpha_soft) * background_value

    All three values are clamped to ``[0, 1]`` to keep downstream LightBurn
    PNG quantisation in range.
    """
    if heightmap.shape != alpha.shape:
        raise ValueError(
            f"heightmap shape {heightmap.shape} does not match alpha {alpha.shape}"
        )
    binary = alpha >= float(binary_threshold)
    if feather_px > 0:
        binary = _binary_dilate(binary, feather_px)
    soft = _gaussian_blur_alpha(binary.astype(np.float32), feather_px)
    soft = np.clip(soft, 0.0, 1.0)
    bg = float(np.clip(background_value, 0.0, 1.0))
    out = soft * heightmap.astype(np.float32) + (1.0 - soft) * bg
    return np.clip(out, 0.0, 1.0).astype(np.float32)
