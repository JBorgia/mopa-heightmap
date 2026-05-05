"""Surface-normal estimators.

Same plug-in pattern as :mod:`zoedepth.laser.backends` and
:mod:`zoedepth.laser.subject_mask`: backends register themselves at import
and the rest of the pipeline asks the registry for a loader by short key.

Default-on, commercially safe options:

* ``finite_diff`` — derive normals from any depth map by central differences.
  Zero dependencies, runs in milliseconds; a fine fallback for offline use
  and the natural pairing for ZoeDepth/DAv2 outputs.
* ``dsine`` — DSINE (Bae & Davison 2024). Apache-2.0. ~150 MB, sharp normals.

Opt-in (non-commercial):

* ``marigold_normals`` — Marigold-Normals (CVPR 2024). CC-BY-NC-4.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "NormalEstimatorSpec",
    "register_estimator",
    "get_estimator",
    "list_estimators",
    "load_estimator",
    "depth_to_normals",
    "DEFAULT_PIXEL_SCALE",
    "EPS_NORM",
    "RGB_HALF",
    "RGB_MAX",
]


# ---------------------------------------------------------------- constants

# Pixel-spacing assumed when converting depth -> normals via finite
# differences. The depth map is unit-less so we use 1.0 (one sample per
# step). Scaling this up exaggerates relief and vice-versa.
DEFAULT_PIXEL_SCALE: float = 1.0

# Numerical floor used when normalising vectors to unit length, to avoid
# 0/0 at perfectly flat regions where the finite-diff gradients vanish.
EPS_NORM: float = 1e-8

# Mid-grey in 8-bit RGB; used for the ``[-1, 1] -> [0, 255]`` normal-map
# encoding consumed by image-space normal viewers and Marigold's RGB output.
RGB_HALF: float = 127.5
RGB_MAX: float = 255.0


# ---------------------------------------------------------------- registry

@dataclass(frozen=True)
class NormalEstimatorSpec:
    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]


_REGISTRY: Dict[str, NormalEstimatorSpec] = {}


def register_estimator(spec: NormalEstimatorSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Normal estimator already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_estimator(key: str) -> NormalEstimatorSpec | None:
    return _REGISTRY.get(key)


def list_estimators(include_opt_in: bool = True) -> Tuple[NormalEstimatorSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_estimator(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No normal estimator registered for: {key!r}")
    return spec.loader(device), device


# ---------------------------------------------------------------- depth->normals

def depth_to_normals(
    depth: np.ndarray,
    *,
    pixel_scale: float = DEFAULT_PIXEL_SCALE,
) -> np.ndarray:
    """Estimate per-pixel unit normals from a depth map by central differences.

    Convention matches :mod:`zoedepth.laser.frankot_chellappa`:

    * ``+x`` points right, ``+y`` points down, ``+z`` points out of the image.
    * Returned normals have shape ``(H, W, 3)`` and unit length.

    The method is exact for piecewise-linear surfaces and well-behaved for
    smooth surfaces; at depth discontinuities it under-estimates Nz, which
    is exactly the behaviour Frankot–Chellappa expects (treats them as
    occluding contour with no constraint).
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D; got shape {depth.shape}")
    z = depth.astype(np.float32)
    # ``np.gradient`` returns (dz/dy, dz/dx) for a 2-D input, in matrix
    # ordering. Divide by pixel spacing so the gradients are in
    # depth-units / world-unit rather than depth-units / pixel.
    dzdy, dzdx = np.gradient(z, float(pixel_scale))
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + EPS_NORM
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return out.astype(np.float32)


# ---------------------------------------------------------------- DSINE wrapper

class _DSINEEstimator:
    """Wrap a DSINE checkpoint exposing ``infer(PIL) -> (H, W, 3)`` normals.

    DSINE returns normals in the camera frame ``(x right, y up, z forward)``;
    we flip the y-axis to match the image-space convention used elsewhere
    in this codebase (``y down``). Output is unit-length float32.
    """

    INFER_SIDE: int = 480  # DSINE's published inference resolution

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device

    def infer(self, image: Image.Image) -> np.ndarray:
        import torch

        rgb = image.convert("RGB")
        target_w, target_h = rgb.size
        # DSINE's pretrained heads are size-flexible but we cap the longest
        # side to keep VRAM bounded; bilinear resample back at the end.
        scale = self.INFER_SIDE / float(max(target_w, target_h))
        new_w = max(1, int(round(target_w * scale)))
        new_h = max(1, int(round(target_h * scale)))
        resized = rgb.resize((new_w, new_h), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / RGB_MAX
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            pred = self._model(tensor)            # (1, 3, H', W') in [-1, 1]
        n = pred[0].detach().to("cpu").float().numpy().transpose(1, 2, 0)
        # Flip y so down is positive (image-space convention).
        n[..., 1] = -n[..., 1]
        # Resample back to source size with bicubic per channel.
        out = np.empty((target_h, target_w, 3), dtype=np.float32)
        for c in range(3):
            ch = Image.fromarray(((n[..., c] + 1.0) * RGB_HALF).astype(np.uint8))
            ch = ch.resize((target_w, target_h), Image.BICUBIC)
            out[..., c] = (np.asarray(ch, dtype=np.float32) / RGB_HALF) - 1.0
        # Re-normalise after resampling.
        norm = np.sqrt((out * out).sum(axis=-1, keepdims=True)) + EPS_NORM
        return (out / norm).astype(np.float32)


def _make_finite_diff_loader(pixel_scale: float) -> Callable[[str], Any]:
    class _FiniteDiff:
        def __init__(self, ps: float) -> None:
            self._ps = ps

        def infer_from_depth(self, depth: np.ndarray) -> np.ndarray:
            return depth_to_normals(depth, pixel_scale=self._ps)

    def _loader(device: str) -> Any:
        del device
        return _FiniteDiff(pixel_scale)

    return _loader


def _make_dsine_loader() -> Callable[[str], Any]:
    def _loader(device: str) -> _DSINEEstimator:
        # Lazy import — DSINE pulls ``torch.hub`` weights only when chosen.
        import torch

        model = torch.hub.load("baegwangbin/DSINE", "DSINE", trust_repo=True)
        model = model.to(device).eval()
        return _DSINEEstimator(model, device)

    return _loader


def _make_marigold_normals_loader() -> Callable[[str], Any]:
    def _loader(device: str) -> Any:
        from diffusers import MarigoldNormalsPipeline

        pipe = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v0-1"
        ).to(device)

        class _MarigoldWrapper:
            def __init__(self, p):
                self._p = p

            def infer(self, image: Image.Image) -> np.ndarray:
                out = self._p(image.convert("RGB"))
                # Marigold returns normals already in image-space [-1, 1].
                return np.asarray(out.prediction, dtype=np.float32)

        return _MarigoldWrapper(pipe)

    return _loader


# Default-on (zero deps / Apache-2.0).
register_estimator(NormalEstimatorSpec(
    key="finite_diff",
    label="Finite differences from depth (offline)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,
    loader=_make_finite_diff_loader(DEFAULT_PIXEL_SCALE),
))
register_estimator(NormalEstimatorSpec(
    key="dsine",
    label="DSINE (Apache-2.0)",
    license="Apache-2.0",
    requires_opt_in=False,
    needs_gpu=True,
    vram_estimate_mb=900,
    loader=_make_dsine_loader(),
))
# Opt-in (CC-BY-NC-4.0).
register_estimator(NormalEstimatorSpec(
    key="marigold_normals",
    label="Marigold Normals (CC-BY-NC-4.0)",
    license="CC-BY-NC-4.0",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=4800,
    loader=_make_marigold_normals_loader(),
))


# Sapiens-Normal-1B (Meta, ECCV 2024) — portrait/human SOTA at 1024 native.
# Replaces the DSINE 480-px ceiling so FC integration of normals can recover
# face/cloth/jewelry detail that DSINE can't. CC-BY-NC-4.0 — opt-in only.

class _SapiensNormalEstimator:
    """Sapiens-Normal TorchScript wrapper exposing ``infer(PIL) -> normals``.

    Output is unit-length float32 (H, W, 3) in image-space convention
    (``+x`` right, ``+y`` down, ``+z`` out of image), matching DSINE's frame.
    """

    INFER_H: int = 1024
    INFER_W: int = 768
    IMAGENET_MEAN: Tuple[float, float, float] = (123.5, 116.5, 103.5)
    IMAGENET_STD: Tuple[float, float, float] = (58.5, 57.0, 57.5)

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device

    def infer(self, image: Image.Image) -> np.ndarray:
        import torch

        rgb = image.convert("RGB")
        target_w, target_h = rgb.size
        resized = rgb.resize((self.INFER_W, self.INFER_H), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32)
        arr = (arr - np.asarray(self.IMAGENET_MEAN, dtype=np.float32)) / \
              np.asarray(self.IMAGENET_STD, dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            pred = self._model(tensor)            # (1, 3, H', W') in [-1, 1]
        n = pred[0].detach().to("cpu").float().numpy().transpose(1, 2, 0)
        # Sapiens convention: +x right, +y up, +z forward. Flip y to image-space.
        n[..., 1] = -n[..., 1]
        # Resample back to source size, per-channel bicubic, then renormalise.
        out = np.empty((target_h, target_w, 3), dtype=np.float32)
        for c in range(3):
            ch = Image.fromarray(((n[..., c] + 1.0) * RGB_HALF).astype(np.uint8))
            ch = ch.resize((target_w, target_h), Image.BICUBIC)
            out[..., c] = (np.asarray(ch, dtype=np.float32) / RGB_HALF) - 1.0
        norm = np.sqrt((out * out).sum(axis=-1, keepdims=True)) + EPS_NORM
        return (out / norm).astype(np.float32)


def _make_sapiens_normal_loader(
    repo: str = "facebook/sapiens-normal-1b-torchscript",
    filename: str = "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
) -> Callable[[str], Any]:
    def _loader(device: str) -> _SapiensNormalEstimator:
        import torch
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(repo_id=repo, filename=filename)
        model = torch.jit.load(ckpt, map_location=device).eval()
        return _SapiensNormalEstimator(model, device)
    return _loader


register_estimator(NormalEstimatorSpec(
    key="sapiens_normal_1b",
    label="Sapiens Normal 1B — portrait SOTA (CC-BY-NC-4.0)",
    license="CC-BY-NC-4.0",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=4500,
    loader=_make_sapiens_normal_loader(),
))
