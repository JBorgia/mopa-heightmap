"""Image delighting — separate albedo from baked-in shadows / highlights.

Photographic depth networks (Depth-Anything-V2, ZoeDepth, Sapiens) are
trained on real photos with their lighting baked in, so they happily
treat a strong specular highlight as a *concavity* (because the highlight
moves with view, like a reflection on a curved surface) and a hard
shadow as raised relief. For laser engraving on metal/jewelry/glossy
subjects, those misreads engrave as visible pits or bumps.

Marigold-IID-Appearance (Bhat et al., 2025) decomposes an image into
albedo + shading + roughness + metallicity, all in one diffusion pass.
Feeding *only the albedo* to the depth backbone removes view-dependent
illumination cues and gives geometry-only depth, which the user then
combines with photo-luminance high-pass for surface texture.

This module is intentionally a thin wrapper that:
    * exposes :func:`load_delighter` returning an instance with an
      ``albedo(image: PIL) -> PIL`` method,
    * lazy-imports diffusers so installs without it just see the stage
      raise a clean ImportError when the user tries to enable it,
    * caches the loaded pipeline so back-to-back renders don't reload
      ~1 GB of weights.

Default backend ``marigold_iid`` is **opt-in (CC-BY-NC-4.0)**, gated on
``allow_nc_weights`` like the other research-licence backends. There is
no commercial-friendly equivalent yet; the stage simply skips when the
user hasn't opted in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "DelighterSpec",
    "register_delighter",
    "get_delighter",
    "list_delighters",
    "load_delighter",
    "DEFAULT_DELIGHTER_KEY",
]


DEFAULT_DELIGHTER_KEY: str = "marigold_iid"


@dataclass(frozen=True)
class DelighterSpec:
    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]


_REGISTRY: Dict[str, DelighterSpec] = {}


def register_delighter(spec: DelighterSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Delighter already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_delighter(key: str) -> DelighterSpec | None:
    return _REGISTRY.get(key)


def list_delighters(include_opt_in: bool = True) -> Tuple[DelighterSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_delighter(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No delighter registered for: {key!r}")
    return spec.loader(device), device


# --------------------------------------------------------- backends

class _MarigoldIIDDelighter:
    """Wrap MarigoldIIDPipeline.albedo to expose a stable ``albedo`` method."""

    def __init__(self, pipeline: Any) -> None:
        self._pipe = pipeline

    def albedo(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        out = self._pipe(rgb)
        # Marigold returns either a PredictionList (newer diffusers) or a
        # tuple/dict; we accept any of the documented shapes.
        prediction = (
            getattr(out, "prediction", None)
            if not isinstance(out, dict) else out.get("prediction")
        )
        if prediction is None:
            prediction = out
        # Coerce to a PIL image at the source resolution.
        arr = np.asarray(prediction, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            albedo_pil = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.shape[-1] == 3:
            albedo_pil = Image.fromarray(arr, mode="RGB")
        else:
            albedo_pil = Image.fromarray(arr[..., :3], mode="RGB")
        if albedo_pil.size != rgb.size:
            albedo_pil = albedo_pil.resize(rgb.size, Image.BICUBIC)
        return albedo_pil


def _make_marigold_iid_loader(
    repo: str = "prs-eth/marigold-iid-appearance-v1-1",
) -> Callable[[str], Any]:
    cache: Dict[str, _MarigoldIIDDelighter] = {}

    def _loader(device: str) -> _MarigoldIIDDelighter:
        if device in cache:
            return cache[device]
        # Lazy import so envs without diffusers still load this module.
        from diffusers import MarigoldIIDAppearancePipeline

        pipe = MarigoldIIDAppearancePipeline.from_pretrained(repo).to(device)
        wrapped = _MarigoldIIDDelighter(pipe)
        cache[device] = wrapped
        return wrapped

    return _loader


register_delighter(DelighterSpec(
    key="marigold_iid",
    label="Marigold-IID-Appearance v1.1 (CC-BY-NC-4.0)",
    license="CC-BY-NC-4.0",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=4800,
    loader=_make_marigold_iid_loader(),
))
