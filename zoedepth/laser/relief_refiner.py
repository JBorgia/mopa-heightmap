"""Bas-relief refinement of a heightmap using the source photo as guide.

Default backend: ``guided-filter`` — He et al. 2010 "Guided Image
Filtering". Edge-preserving smoothing of the heightmap *guided* by the
luminance of the source photo, so depth edges snap to photo edges and
flat regions stay flat. Pure NumPy.

Opt-in backend: ``controlnet-depth-bas-relief`` — runs the photo +
heightmap through a ControlNet-Depth model fine-tuned on bas-relief
references for a stylised pass. Stubbed; requires ``diffusers`` and a
fine-tuned LoRA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "ReliefRefinerSpec",
    "register_refiner",
    "get_refiner",
    "list_refiners",
    "load_refiner",
    "guided_filter",
    "DEFAULT_REFINER_KEY",
    "DEFAULT_REFINE_STRENGTH",
    "DEFAULT_GUIDED_FILTER_RADIUS",
    "DEFAULT_GUIDED_FILTER_EPS",
    "REFINE_STRENGTH_MIN",
    "REFINE_STRENGTH_MAX",
]


# ----------------------------------------------------------- constants

DEFAULT_REFINER_KEY: str = "guided-filter"

# Blend factor between the raw heightmap (0.0) and the refined heightmap
# (1.0). Defaults to half-strength so we improve edges without losing the
# depth network's confident regions.
DEFAULT_REFINE_STRENGTH: float = 0.5
REFINE_STRENGTH_MIN: float = 0.0
REFINE_STRENGTH_MAX: float = 1.0

# Radius (in pixels) of the local window the guided filter averages over.
# 8 px ≈ a full eye/cheekbone on a 768 px portrait — keeps facial features
# sharp without smearing them into the background.
DEFAULT_GUIDED_FILTER_RADIUS: int = 8

# Regularisation: smaller = more edge-preserving (looks like a stylised
# bas-relief), larger = closer to plain Gaussian smoothing.
DEFAULT_GUIDED_FILTER_EPS: float = 1e-3

_EPS_DIV: float = 1e-8


# ----------------------------------------------------------- guided filter

def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter over a (2r+1)×(2r+1) window via cumulative sums."""
    if radius <= 0:
        return arr.astype(np.float32, copy=True)
    a = arr.astype(np.float32, copy=False)
    h, w = a.shape
    csum = np.zeros((h + 1, w + 1), dtype=np.float32)
    csum[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    out = np.empty_like(a)
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            area = (y1 - y0) * (x1 - x0)
            out[y, x] = (
                csum[y1, x1] - csum[y0, x1] - csum[y1, x0] + csum[y0, x0]
            ) / max(area, 1)
    return out


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    *,
    radius: int = DEFAULT_GUIDED_FILTER_RADIUS,
    eps: float = DEFAULT_GUIDED_FILTER_EPS,
) -> np.ndarray:
    """Edge-preserving smoothing of ``src`` guided by ``guide``.

    Both inputs must be 2-D float arrays of the same shape, in ``[0, 1]``.
    Returns the filtered ``src`` (same shape, float32).
    """
    if guide.shape != src.shape:
        raise ValueError(
            f"guide and src must have the same shape; got {guide.shape} vs {src.shape}"
        )
    if guide.ndim != 2:
        raise ValueError("guided_filter currently supports 2-D guides only")
    g = guide.astype(np.float32, copy=False)
    s = src.astype(np.float32, copy=False)
    mean_g = _box_blur(g, radius)
    mean_s = _box_blur(s, radius)
    mean_gs = _box_blur(g * s, radius)
    cov_gs = mean_gs - mean_g * mean_s
    mean_gg = _box_blur(g * g, radius)
    var_g = mean_gg - mean_g * mean_g
    a = cov_gs / (var_g + float(eps))
    b = mean_s - a * mean_g
    mean_a = _box_blur(a, radius)
    mean_b = _box_blur(b, radius)
    return (mean_a * g + mean_b).astype(np.float32, copy=False)


# ----------------------------------------------------------- registry

@dataclass(frozen=True)
class ReliefRefinerSpec:
    """Metadata + loader for one bas-relief refinement backend."""

    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]      # device -> object with .refine(image, heightmap, strength)


_REGISTRY: Dict[str, ReliefRefinerSpec] = {}


def register_refiner(spec: ReliefRefinerSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Relief refiner already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_refiner(key: str) -> ReliefRefinerSpec | None:
    return _REGISTRY.get(key)


def list_refiners(include_opt_in: bool = True) -> Tuple[ReliefRefinerSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_refiner(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No relief refiner registered for: {key!r}")
    return spec.loader(device), device


# ----------------------------------------------------------- default backends

class _GuidedFilterRefiner:
    """Apply :func:`guided_filter` then blend with the original heightmap."""

    def __init__(self, radius: int, eps: float) -> None:
        self._radius = int(radius)
        self._eps = float(eps)

    def refine(
        self,
        image: Image.Image,
        heightmap: np.ndarray,
        strength: float = DEFAULT_REFINE_STRENGTH,
    ) -> np.ndarray:
        if heightmap.ndim != 2:
            raise ValueError(f"heightmap must be 2-D, got shape {heightmap.shape}")
        guide_pil = image.convert("L").resize(
            (heightmap.shape[1], heightmap.shape[0]), Image.BILINEAR,
        )
        guide = np.asarray(guide_pil, dtype=np.float32) / 255.0
        filtered = guided_filter(guide, heightmap.astype(np.float32, copy=False),
                                  radius=self._radius, eps=self._eps)
        s = float(np.clip(strength, REFINE_STRENGTH_MIN, REFINE_STRENGTH_MAX))
        return ((1.0 - s) * heightmap.astype(np.float32, copy=False)
                + s * filtered).astype(np.float32, copy=False)


class _ControlNetReliefStub:
    """Loader-time guard for the ControlNet bas-relief opt-in backend."""

    def __init__(self, device: str) -> None:
        self._device = device
        try:
            import diffusers  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ControlNet relief refiner is opt-in: pip install diffusers "
                "transformers accelerate to enable this backend."
            ) from exc

    def refine(self, image, heightmap, strength=DEFAULT_REFINE_STRENGTH):
        raise RuntimeError(
            "ControlNet bas-relief inference not wired in this build; "
            "install diffusers and replace _ControlNetReliefStub with the "
            "upstream pipeline."
        )


def _make_guided_loader(radius: int, eps: float) -> Callable[[str], Any]:
    def _load(_device: str) -> Any:
        return _GuidedFilterRefiner(radius=radius, eps=eps)
    return _load


# ----------------------------------------------------------- registrations

register_refiner(ReliefRefinerSpec(
    key="guided-filter",
    label="Guided filter (CPU, instant)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,
    loader=_make_guided_loader(
        radius=DEFAULT_GUIDED_FILTER_RADIUS,
        eps=DEFAULT_GUIDED_FILTER_EPS,
    ),
))


register_refiner(ReliefRefinerSpec(
    key="controlnet-depth-bas-relief",
    label="ControlNet-Depth bas-relief LoRA (GPU, opt-in)",
    license="OpenRAIL-M",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=6000,
    loader=lambda device: _ControlNetReliefStub(device),
))
