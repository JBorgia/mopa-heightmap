"""Pluggable depth-inference backends.

Each backend exposes the same minimal interface ZoeDepth's models expose to
:class:`HeightmapService` — an ``infer_pil(image, pad_input=True,
with_flip_aug=True) -> np.ndarray`` method returning a 2-D float32 depth map
in *ZoeDepth-style metric semantics* (``larger value == farther from camera``).

Backends are registered by short string keys (e.g. ``"DAv2_Base"``) that the UI
exposes in the model dropdown. The default ZoeDepth variants
(``ZoeD_NK`` / ``ZoeD_N`` / ``ZoeD_K``) are *not* registered here — they keep
loading through ``hubconf`` for backward compatibility. The service falls back
to hubconf when a name is missing from this registry.

License policy (see ``IMPLEMENTATION_PLAN.md`` §11): only Apache-2.0 / MIT
weights ship as defaults. Non-commercial weights (DAv2-Large, RMBG-2.0,
Sapiens, Marigold, Hunyuan3D-2) require an explicit opt-in toggle in
Settings before they are downloaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "BackendSpec",
    "register_backend",
    "get_backend",
    "list_backends",
    "load_backend",
    "DepthAnythingV2Wrapper",
]


@dataclass(frozen=True)
class BackendSpec:
    """Metadata + loader for one depth backend."""

    key: str                             # short id, used by UI + cache key
    label: str                           # human-readable label for dropdown
    license: str                         # SPDX-ish identifier
    requires_opt_in: bool                # True for non-commercial weights
    vram_estimate_mb: int                # rough fp16 working set
    loader: Callable[[str], Any]         # device -> model with .infer_pil(...)


_REGISTRY: Dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Backend already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_backend(key: str) -> BackendSpec | None:
    return _REGISTRY.get(key)


def list_backends(include_opt_in: bool = True) -> Tuple[BackendSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_backend(key: str, device: str) -> Tuple[Any, str]:
    """Instantiate a backend by key.

    Returns ``(model, device)`` matching ``HeightmapService``'s loader contract.
    Raises ``KeyError`` if the key is not registered.
    """
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No backend registered for: {key!r}")
    model = spec.loader(device)
    return model, device


# ---------------------------------------------------------------------------
# Depth-Anything-V2 (Apache-2.0 — Base/Small ship as defaults; Large is opt-in)
# ---------------------------------------------------------------------------

# HF Hub IDs for the official Depth-Anything-V2 checkpoints. ``Hf`` variants
# include the transformers config so we can load them via AutoModel.
_DAV2_REPO = {
    "Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2Wrapper:
    """Adapter that gives a HuggingFace Depth-Anything-V2 model the
    ``infer_pil`` shape that :class:`HeightmapService` expects.

    The HF model returns ``predicted_depth`` as an *affine-invariant inverse
    depth* (larger = closer). We invert with ``pred.max() - pred`` so the
    output follows ZoeDepth's convention (``larger = farther``) and the
    existing percentile / ``black_is_deep`` logic works without changes.
    """

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def _flip_aug(self, image: Image.Image) -> Image.Image:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def _infer_one(self, image: Image.Image) -> np.ndarray:
        import torch

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.inference_mode():
            outputs = self._model(pixel_values=pixel_values)
        # Returns a tensor of shape (1, H', W'); resample to source size.
        pred = outputs.predicted_depth  # (1, H', W')
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)    # (1, 1, H', W')
        target_size = (image.size[1], image.size[0])  # (H, W)
        pred = torch.nn.functional.interpolate(
            pred, size=target_size, mode="bicubic", align_corners=False
        )
        # Drop only the batch+channel dims so we keep 2-D shape even when
        # one of H/W is 1 (avoids np.fliplr / shape assertions tripping).
        depth = pred[0, 0].detach().to("cpu").float().numpy()
        # Flip from inverse-depth-like semantics to metric-like (larger=farther).
        pmax = float(depth.max())
        return (pmax - depth).astype(np.float32)

    def infer_pil(
        self,
        image: Image.Image,
        pad_input: bool = True,                # accepted for API parity, unused
        with_flip_aug: bool = True,
    ) -> np.ndarray:
        del pad_input  # DAv2's processor handles its own resize/pad
        depth = self._infer_one(image)
        if with_flip_aug:
            mirrored = self._infer_one(self._flip_aug(image))
            depth = 0.5 * (depth + np.fliplr(mirrored).astype(np.float32))
        return depth


def _make_dav2_loader(size: str) -> Callable[[str], Any]:
    """Return a loader that lazily instantiates a Depth-Anything-V2 backend."""

    def _loader(device: str) -> DepthAnythingV2Wrapper:
        # Lazy import — transformers is heavy and not needed if the user
        # never selects this backend.
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = _DAV2_REPO[size]
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()
        return DepthAnythingV2Wrapper(model, processor, device)

    return _loader


# Default-on (Apache-2.0): Small (~25M) and Base (~97M).
register_backend(BackendSpec(
    key="DAv2_Small",
    label="Depth-Anything-V2 Small (Apache-2.0)",
    license="Apache-2.0",
    requires_opt_in=False,
    vram_estimate_mb=400,
    loader=_make_dav2_loader("Small"),
))
register_backend(BackendSpec(
    key="DAv2_Base",
    label="Depth-Anything-V2 Base (Apache-2.0)",
    license="Apache-2.0",
    requires_opt_in=False,
    vram_estimate_mb=900,
    loader=_make_dav2_loader("Base"),
))
# Opt-in only (CC-BY-NC-4.0 weights).
register_backend(BackendSpec(
    key="DAv2_Large",
    label="Depth-Anything-V2 Large (CC-BY-NC-4.0)",
    license="CC-BY-NC-4.0",
    requires_opt_in=True,
    vram_estimate_mb=2200,
    loader=_make_dav2_loader("Large"),
))


# ---------------------------------------------------------------------------
# Sapiens-Depth-1B (Meta, ECCV 2024) — portrait/human-only SOTA depth.
# Trained on 500K synthetic photogrammetry humans at 1024×768; +22.4 % RMSE
# on Hi4D over prior SOTA. CC-BY-NC-4.0 — opt-in only.
# Distributed as TorchScript (.pt2) via HuggingFace; loaded with torch.jit.
# ---------------------------------------------------------------------------

class SapiensDepthWrapper:
    """Adapter that gives Sapiens-Depth a ZoeDepth-style ``infer_pil`` API.

    Sapiens emits a height field where larger value = closer to camera; we
    invert to match the ZoeDepth convention (larger = farther) so the
    existing percentile / orient / black_is_deep logic works unchanged.
    """

    INFER_H: int = 1024
    INFER_W: int = 768
    IMAGENET_MEAN: Tuple[float, float, float] = (123.5, 116.5, 103.5)
    IMAGENET_STD: Tuple[float, float, float] = (58.5, 57.0, 57.5)

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device

    def _preprocess(self, image: Image.Image):
        import torch

        rgb = image.convert("RGB").resize((self.INFER_W, self.INFER_H), Image.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32)
        arr = (arr - np.asarray(self.IMAGENET_MEAN, dtype=np.float32)) / \
              np.asarray(self.IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

    def infer_pil(
        self,
        image: Image.Image,
        pad_input: bool = True,
        with_flip_aug: bool = True,
    ) -> np.ndarray:
        del pad_input  # Sapiens has its own fixed input shape
        import torch

        with torch.inference_mode():
            pred = self._model(self._preprocess(image))
        if pred.dim() == 4:
            pred = pred[:, 0]                         # (1, H', W')
        target_size = (image.size[1], image.size[0])  # (H, W)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False,
        )
        depth = pred[0, 0].detach().to("cpu").float().numpy()
        # Sapiens "depth" is closeness; flip to ZoeDepth convention.
        depth = float(depth.max()) - depth
        if with_flip_aug:
            mirrored = self._infer_one_flipped(image)
            depth = 0.5 * (depth + mirrored)
        return depth.astype(np.float32)

    def _infer_one_flipped(self, image: Image.Image) -> np.ndarray:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        single_pass = self.__class__(self._model, self._device)
        out = single_pass.infer_pil(flipped, with_flip_aug=False)
        return np.fliplr(out)


def _make_sapiens_depth_loader(repo: str = "facebook/sapiens-depth-1b-torchscript") -> Callable[[str], Any]:
    def _loader(device: str) -> SapiensDepthWrapper:
        # Lazy: huggingface_hub only imported when this backend is selected.
        import torch
        from huggingface_hub import hf_hub_download

        # The HF repo ships several quantisations; the bf16 TorchScript is
        # the right balance of size and quality for our use case.
        ckpt_path = hf_hub_download(
            repo_id=repo,
            filename="sapiens_1b_render_people_epoch_88_torchscript.pt2",
        )
        model = torch.jit.load(ckpt_path, map_location=device).eval()
        return SapiensDepthWrapper(model, device)
    return _loader


register_backend(BackendSpec(
    key="Sapiens_Depth_1B",
    label="Sapiens Depth 1B — portrait SOTA (CC-BY-NC-4.0)",
    license="CC-BY-NC-4.0",
    requires_opt_in=True,
    vram_estimate_mb=4500,
    loader=_make_sapiens_depth_loader(),
))


# ---------------------------------------------------------------------------
# TripoSR (Stability AI, 2024) — image → 3D mesh → orthographic Z heightmap.
# Different model class entirely from the depth networks above: instead of
# regressing a 2-D depth field, it generates a full triplane 3-D
# representation, marching-cubes-extracts a mesh, and we render the
# front-facing Z buffer. The mesh prior gives sculptural confidence
# (separate fingers, defined cheekbones, fur strands) that monocular
# depth networks lack — see ``feedback_sculptok_parity_ceiling.md``.
# ---------------------------------------------------------------------------

def _make_triposr_loader() -> Callable[[str], Any]:
    def _loader(device: str) -> Any:
        from .mesh_depth import load_triposr_depth_backend
        return load_triposr_depth_backend(device)
    return _loader


register_backend(BackendSpec(
    key="TripoSR",
    label="TripoSR — image → 3D mesh → ortho Z (MIT)",
    license="MIT",
    requires_opt_in=False,
    vram_estimate_mb=2500,
    loader=_make_triposr_loader(),
))
