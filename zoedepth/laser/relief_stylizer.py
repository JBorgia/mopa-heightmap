"""ControlNet-Depth bas-relief stylization.

Takes a photo + the depth map our pipeline already computed, runs them
through ControlNet-Depth conditioning on a diffusion model with a
bas-relief prompt, and produces a *stylized bas-relief render of the
input subject*. Downstream code (depth estimator) then runs on the
stylized image to recover relief-style detail (fur strands, beard hair,
embroidery weave) that monocular depth on the original photo can't see.

Why this works:
    Monocular depth networks like DAv2 are trained on natural scenes and
    portraits, not on bas-relief sculpture. They produce smooth,
    photographically-plausible depth — which is the wrong domain for
    laser-engraved relief that wants high-frequency surface texture.
    A diffusion model conditioned on the input depth + a bas-relief
    prompt hallucinates the *right kind of detail* for each region:
    fur strands on shoulders, beard hair, fabric weave on cloth,
    sculpted facial planes. We then re-estimate depth on that
    hallucinated-but-coherent bas-relief image and get a heightmap with
    the texture our reference target ("sculptok") has.

Default backend: SDXL base + ``diffusers/controlnet-depth-sdxl-1.0``.
Both are CC-BY-SA / OpenRAIL-M (commercially usable) and the canonical
ControlNet-Depth path. Lazy-loaded; weights pull on first use to
``~/.cache/huggingface/hub`` (~12 GB total).

Settings flow through :mod:`zoedepth.laser.service` as
``relief_stylize_*`` keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "RELIEF_STYLE_PROMPT",
    "RELIEF_NEGATIVE_PROMPT",
    "DEFAULT_BACKEND_KEY",
    "DEFAULT_INFERENCE_STEPS",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_CONTROLNET_STRENGTH",
    "ReliefStylizerSpec",
    "register_stylizer",
    "get_stylizer",
    "list_stylizers",
    "load_stylizer",
    "depth_to_controlnet_pil",
]


# Prompt tuned for "marble bas-relief sculpture" output. The combination
# of "marble" + "stone" + "deep shadow" + "intricate detail" pushes the
# diffusion model toward the sculptok-style relief aesthetic without
# specifying a particular subject.
RELIEF_STYLE_PROMPT: str = (
    "marble bas-relief sculpture, classical relief carving, white stone, "
    "deep shadow, museum quality, intricate sculptural detail, "
    "high-frequency surface texture, refined sculpted geometry, "
    "8k, masterpiece"
)

# Negative prompt steers away from photorealistic / painterly outputs
# that would re-introduce the same domain mismatch we're trying to escape.
RELIEF_NEGATIVE_PROMPT: str = (
    "color, painting, photograph, blurry, flat, smooth, low detail, "
    "drawing, sketch, anime, cartoon, jpeg artifacts, watermark"
)

DEFAULT_BACKEND_KEY: str = "sd15_controlnet_depth"

# Sensible defaults for SDXL inference.
DEFAULT_INFERENCE_STEPS: int = 25
DEFAULT_GUIDANCE_SCALE: float = 7.5
DEFAULT_CONTROLNET_STRENGTH: float = 0.95


@dataclass(frozen=True)
class ReliefStylizerSpec:
    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]


_REGISTRY: Dict[str, ReliefStylizerSpec] = {}


def register_stylizer(spec: ReliefStylizerSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Relief stylizer already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_stylizer(key: str) -> ReliefStylizerSpec | None:
    return _REGISTRY.get(key)


def list_stylizers(include_opt_in: bool = True) -> Tuple[ReliefStylizerSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_stylizer(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No relief stylizer registered for: {key!r}")
    return spec.loader(device), device


# --------------------------------------------------------- helpers

def depth_to_controlnet_pil(
    depth: np.ndarray,
    *,
    target_size: Tuple[int, int] | None = None,
) -> Image.Image:
    """Convert a 2-D depth map (any scale) to the RGB depth image
    ControlNet-Depth expects.

    ControlNet-Depth was trained on MiDaS-style depth maps where ``brighter
    = closer``. Our DAv2 backend already inverts to that convention, but
    upstream depth could be either polarity — we percentile-normalise to
    ``[0, 1]`` and let the user invert via the ``relief_stylize_invert``
    setting if a particular backend produces the opposite sign.
    """
    arr = np.asarray(depth, dtype=np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    span = max(p99 - p1, 1e-6)
    norm = np.clip((arr - p1) / span, 0.0, 1.0)
    # ControlNet-Depth expects "brighter = closer", which is the
    # *inverse* of ZoeDepth/DAv2's "larger = farther" output. Flip.
    norm = 1.0 - norm
    rgb = (norm[..., None].repeat(3, axis=-1) * 255).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    if target_size is not None and target_size != pil.size:
        pil = pil.resize(target_size, Image.BICUBIC)
    return pil


# --------------------------------------------------------- backends

class _SDXLControlNetReliefStylizer:
    """SDXL + ControlNet-Depth + bas-relief prompt."""

    BASE_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_MODEL: str = "diffusers/controlnet-depth-sdxl-1.0"
    # SDXL native resolution. Output gets resampled back to the source
    # input size by the caller.
    NATIVE_SIDE: int = 1024

    def __init__(self, pipeline: Any, device: str) -> None:
        self._pipe = pipeline
        self._device = device

    def stylize(
        self,
        image: Image.Image,
        depth: np.ndarray,
        *,
        prompt: str = RELIEF_STYLE_PROMPT,
        negative_prompt: str = RELIEF_NEGATIVE_PROMPT,
        steps: int = DEFAULT_INFERENCE_STEPS,
        guidance: float = DEFAULT_GUIDANCE_SCALE,
        controlnet_strength: float = DEFAULT_CONTROLNET_STRENGTH,
        seed: int | None = None,
    ) -> Image.Image:
        import torch

        target_w, target_h = image.size
        # SDXL likes square multiples of 8; aim for the closest.
        infer_long = self.NATIVE_SIDE
        scale = infer_long / float(max(target_w, target_h))
        infer_w = max(8, int(round(target_w * scale / 8.0)) * 8)
        infer_h = max(8, int(round(target_h * scale / 8.0)) * 8)

        depth_pil = depth_to_controlnet_pil(depth, target_size=(infer_w, infer_h))

        generator = (
            torch.Generator(device=self._device).manual_seed(int(seed))
            if seed is not None else None
        )
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_pil,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(controlnet_strength),
            width=infer_w,
            height=infer_h,
            generator=generator,
        )
        out = result.images[0]
        if out.size != (target_w, target_h):
            out = out.resize((target_w, target_h), Image.BICUBIC)
        return out


def _make_sdxl_controlnet_loader() -> Callable[[str], Any]:
    cache: Dict[str, _SDXLControlNetReliefStylizer] = {}

    def _loader(device: str) -> _SDXLControlNetReliefStylizer:
        if device in cache:
            return cache[device]
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            AutoencoderKL,
        )

        # fp16 only when CUDA — CPU inference is feasible (slow) and uses
        # fp32 for stability.
        is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        dtype = torch.float16 if is_cuda else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            _SDXLControlNetReliefStylizer.CONTROLNET_MODEL, torch_dtype=dtype,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            _SDXLControlNetReliefStylizer.BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16" if is_cuda else None,
            use_safetensors=True,
        )
        pipe = pipe.to(device)
        if is_cuda:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        wrapped = _SDXLControlNetReliefStylizer(pipe, device)
        cache[device] = wrapped
        return wrapped

    return _loader


register_stylizer(ReliefStylizerSpec(
    key="sdxl_controlnet_depth",
    label="SDXL base + ControlNet-Depth (CreativeML Open RAIL-M, 8 GB VRAM)",
    license="OpenRAIL-M",
    requires_opt_in=False,
    needs_gpu=True,
    vram_estimate_mb=8000,
    loader=_make_sdxl_controlnet_loader(),
))


# ---------------------------------------------------------- SD1.5 backend
# Smaller than SDXL — fits in ~3.5 GB VRAM at fp16 with model_cpu_offload.
# Quality on bas-relief style is comparable to SDXL because the prompt +
# ControlNet-Depth carry most of the signal, not the base model's
# photorealism. Default backend on consumer GPUs (≤ 6 GB).

class _SD15ControlNetReliefStylizer:
    """SD1.5 + ControlNet-Depth-v1.1 + bas-relief prompt.

    Optimised for 4 GB VRAM cards (Quadro P2000, GTX 1650, etc.) via
    ``enable_model_cpu_offload`` and fp16. Inference resolution defaults
    to 768 × 768 because SD1.5 was trained on 512² and gracefully scales
    up to ~768²; beyond that the output starts to repeat (well-known
    SD1.5 limitation).
    """

    BASE_MODEL: str = "Lykon/dreamshaper-8"   # SD1.5 finetune, good at stylised outputs
    CONTROLNET_MODEL: str = "lllyasviel/control_v11f1p_sd15_depth"
    NATIVE_SIDE: int = 768

    def __init__(self, pipeline: Any, device: str) -> None:
        self._pipe = pipeline
        self._device = device

    def stylize(
        self,
        image: Image.Image,
        depth: np.ndarray,
        *,
        prompt: str = RELIEF_STYLE_PROMPT,
        negative_prompt: str = RELIEF_NEGATIVE_PROMPT,
        steps: int = DEFAULT_INFERENCE_STEPS,
        guidance: float = DEFAULT_GUIDANCE_SCALE,
        controlnet_strength: float = DEFAULT_CONTROLNET_STRENGTH,
        seed: int | None = None,
    ) -> Image.Image:
        import torch

        target_w, target_h = image.size
        infer_long = self.NATIVE_SIDE
        scale = infer_long / float(max(target_w, target_h))
        # SD1.5 expects multiples of 8.
        infer_w = max(8, int(round(target_w * scale / 8.0)) * 8)
        infer_h = max(8, int(round(target_h * scale / 8.0)) * 8)

        depth_pil = depth_to_controlnet_pil(depth, target_size=(infer_w, infer_h))

        # The pipeline's generator must be on the device the unet runs on.
        # When ``enable_model_cpu_offload`` is active, that's CPU until the
        # unet step; ``Generator(device='cpu')`` is the safe choice.
        generator = (
            torch.Generator(device="cpu").manual_seed(int(seed))
            if seed is not None else None
        )
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_pil,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(controlnet_strength),
            width=infer_w,
            height=infer_h,
            generator=generator,
        )
        out = result.images[0]
        if out.size != (target_w, target_h):
            out = out.resize((target_w, target_h), Image.BICUBIC)
        return out


def _make_sd15_controlnet_loader() -> Callable[[str], Any]:
    cache: Dict[str, _SD15ControlNetReliefStylizer] = {}

    def _loader(device: str) -> _SD15ControlNetReliefStylizer:
        if device in cache:
            return cache[device]
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        dtype = torch.float16 if is_cuda else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            _SD15ControlNetReliefStylizer.CONTROLNET_MODEL, torch_dtype=dtype,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            _SD15ControlNetReliefStylizer.BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,            # bas-relief sculptures aren't NSFW; saves ~1 GB
            requires_safety_checker=False,
            use_safetensors=True,
        )
        # UniPC scheduler converges in fewer steps than DDIM/DPM — relevant
        # on a 4 GB card where every step counts.
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        if is_cuda:
            # CPU-offload swaps modules in/out so a 4 GB card can run an
            # SD1.5 + ControlNet pipeline that wouldn't fit otherwise.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        wrapped = _SD15ControlNetReliefStylizer(pipe, device)
        cache[device] = wrapped
        return wrapped

    return _loader


register_stylizer(ReliefStylizerSpec(
    key="sd15_controlnet_depth",
    label="SD1.5 (Dreamshaper-8) + ControlNet-Depth v1.1 (CreativeML Open RAIL-M, 4 GB VRAM)",
    license="OpenRAIL-M",
    requires_opt_in=False,
    needs_gpu=True,
    vram_estimate_mb=3500,
    loader=_make_sd15_controlnet_loader(),
))
