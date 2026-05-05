"""Super-resolution preprocessing for low-quality input photos.

Most engraving-quality losses come from low-resolution source images: a
512-px selfie produces a 512-px depth map, and the polish pass has nothing
real to engrave at higher detail. This stage upscales sub-threshold images
*before* the depth network sees them so the entire pipeline benefits.

Permissive default: ``lanczos`` resample (zero deps, instant). Opt-in
heavy default: ``realesrgan-x4plus`` (BSD-3, 17 M params, requires the
``realesrgan`` package).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from PIL import Image


__all__ = [
    "SuperResolverSpec",
    "register_resolver",
    "get_resolver",
    "list_resolvers",
    "load_resolver",
    "auto_upscale",
    "DEFAULT_RESOLVER_KEY",
    "DEFAULT_TARGET_LONG_SIDE_PX",
    "MAX_AUTO_UPSCALE_FACTOR",
    "MIN_AUTO_UPSCALE_FACTOR",
]


# ----------------------------------------------------------- constants

# Permissive default backend used when nothing else is specified.
DEFAULT_RESOLVER_KEY: str = "lanczos"

# Below this longest-side resolution we trigger an auto-upscale by default.
# 1024 is a sweet spot: depth networks still extract good geometry, and the
# polish pass has enough pixels to feed a 200 mm engraving at ~150 dpi.
DEFAULT_TARGET_LONG_SIDE_PX: int = 1024

# Hard ceiling on auto-upscale; beyond 4× even Real-ESRGAN starts inventing
# texture that doesn't help the depth network.
MAX_AUTO_UPSCALE_FACTOR: float = 4.0

# Floor under which we don't bother (1.0 == no-op).
MIN_AUTO_UPSCALE_FACTOR: float = 1.0


# ----------------------------------------------------------- registry

@dataclass(frozen=True)
class SuperResolverSpec:
    """Metadata + loader for one super-resolution backend."""

    key: str
    label: str
    license: str
    requires_opt_in: bool
    max_scale: float
    vram_estimate_mb: int
    loader: Callable[[str], Any]    # device -> object with .upscale(PIL, scale) -> PIL


_REGISTRY: Dict[str, SuperResolverSpec] = {}


def register_resolver(spec: SuperResolverSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Super-resolver already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_resolver(key: str) -> SuperResolverSpec | None:
    return _REGISTRY.get(key)


def list_resolvers(include_opt_in: bool = True) -> Tuple[SuperResolverSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_resolver(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No super-resolver registered for: {key!r}")
    return spec.loader(device), device


# ----------------------------------------------------------- default backends

class _PILResampler:
    """Pure-PIL upscaler used as the zero-dep default."""

    def __init__(self, resample: int) -> None:
        self._resample = resample

    def upscale(self, image: Image.Image, scale: float) -> Image.Image:
        if scale <= 1.0:
            return image
        w, h = image.size
        return image.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            self._resample,
        )


class _RealESRGANResolver:
    """Real upscaler that wraps :class:`realesrgan.RealESRGANer`.

    Loads the official ``RealESRGAN_x4plus`` weights from the upstream
    GitHub release on first use, caches them under the user's torch hub
    cache, and runs inference with tile-based memory bounding so this
    works on a 4 GB GPU. Falls back to CPU when no GPU is available.
    """

    # Upstream-published weights URL. Stable since 2021; same hash everyone
    # else points to. ~17 MB.
    WEIGHT_URL: str = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    )
    WEIGHT_FILE: str = "RealESRGAN_x4plus.pth"
    SCALE_NATIVE: int = 4

    def __init__(self, device: str) -> None:
        self._device = device
        self._engine = self._build_engine(device)

    @classmethod
    def _build_engine(cls, device: str):
        from pathlib import Path
        from urllib.request import urlretrieve
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        cache_dir = Path.home() / ".cache" / "mopa-heightmap" / "realesrgan"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / cls.WEIGHT_FILE
        if not weights_path.exists():
            urlretrieve(cls.WEIGHT_URL, weights_path)

        net = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=cls.SCALE_NATIVE,
        )
        is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        return RealESRGANer(
            scale=cls.SCALE_NATIVE,
            model_path=str(weights_path),
            model=net,
            tile=512 if is_cuda else 256,   # tile keeps VRAM bounded
            tile_pad=10,
            pre_pad=0,
            half=is_cuda,                    # fp16 only on CUDA
            device=device if is_cuda else "cpu",
        )

    def upscale(self, image: Image.Image, scale: float) -> Image.Image:
        if scale <= 1.0:
            return image
        # Real-ESRGAN's enhance() takes a numpy BGR array and an outscale
        # factor. We respect the requested scale up to the network's 4×
        # native (it's bicubic above that, which we'd rather avoid; the
        # auto_upscale clamp prevents outscale > 4 anyway).
        import cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_bgr, _ = self._engine.enhance(bgr, outscale=float(scale))
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)


def _make_realesrgan_loader() -> Callable[[str], Any]:
    cache: Dict[str, _RealESRGANResolver] = {}

    def _load(device: str) -> _RealESRGANResolver:
        if device in cache:
            return cache[device]
        resolver = _RealESRGANResolver(device)
        cache[device] = resolver
        return resolver

    return _load


def _make_pil_loader(resample: int) -> Callable[[str], Any]:
    def _load(_device: str) -> Any:
        return _PILResampler(resample)
    return _load


# ----------------------------------------------------------- public helpers

def auto_upscale(
    image: Image.Image,
    *,
    target_long_side: int = DEFAULT_TARGET_LONG_SIDE_PX,
    resolver_key: str = DEFAULT_RESOLVER_KEY,
    device: str = "cpu",
) -> Image.Image:
    """Upscale ``image`` to at least ``target_long_side`` on its longest edge.

    Returns the image unchanged when it's already large enough or when the
    required scale exceeds :data:`MAX_AUTO_UPSCALE_FACTOR` (in which case
    upscaling would invent more than it preserves).
    """
    if target_long_side <= 0:
        raise ValueError("target_long_side must be positive")
    longest = max(image.size)
    if longest >= target_long_side:
        return image
    scale = target_long_side / float(longest)
    if scale <= MIN_AUTO_UPSCALE_FACTOR:
        return image
    if scale > MAX_AUTO_UPSCALE_FACTOR:
        scale = MAX_AUTO_UPSCALE_FACTOR
    resolver, _ = load_resolver(resolver_key, device)
    return resolver.upscale(image, scale)


# ----------------------------------------------------------- registrations

register_resolver(SuperResolverSpec(
    key="lanczos",
    label="Lanczos resample (CPU, instant)",
    license="MIT",
    requires_opt_in=False,
    max_scale=MAX_AUTO_UPSCALE_FACTOR,
    vram_estimate_mb=0,
    loader=_make_pil_loader(Image.LANCZOS),
))


register_resolver(SuperResolverSpec(
    key="bicubic",
    label="Bicubic resample (CPU, instant)",
    license="MIT",
    requires_opt_in=False,
    max_scale=MAX_AUTO_UPSCALE_FACTOR,
    vram_estimate_mb=0,
    loader=_make_pil_loader(Image.BICUBIC),
))


register_resolver(SuperResolverSpec(
    key="realesrgan-x4plus",
    label="Real-ESRGAN x4plus (BSD-3-Clause)",
    license="BSD-3-Clause",
    requires_opt_in=False,             # weights + package are commercial-OK
    max_scale=4.0,
    vram_estimate_mb=2048,
    loader=_make_realesrgan_loader(),
))
