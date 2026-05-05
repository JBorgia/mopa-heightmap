"""HeightmapService: single orchestrator powering both CLI and UI.

Pipeline (v1):
    image (PIL)
        -> EXIF transpose
        -> depth inference (cached by input hash + model + inference flags)
        -> heightmap shaping (Stage C)
        -> preview rendering
        -> export bundle (atomic writes, naming policy, sidecar JSON)

Higher-stage processing (Stage A input conditioning, Stage B ensemble, Stage D
multi-pass layer derivation) plugs in here in later phases without changing
either the CLI or the UI.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
import torch
from PIL import Image, ImageOps

from . import exporter as _exporter
from .heightmap import process_depth_to_heightmap
from .history import append_history, make_entry
from .imgproc import condition_input, settings_from_mapping
from .preview import create_calibration_ramp, render_preview
from .profiles import load_profile
from .settings import AppSettings, resolve_device, resolve_precision
from .tiling import infer_tiled_pil


DEFAULT_SETTINGS: Dict[str, Any] = {
    "near_percentile": 5.0,
    "far_percentile": 95.0,
    "gamma": 0.72,
    "contrast": 1.0,
    "midtone_boost": 0.0,
    "deep_limit": 0.04,
    "surface_limit": 0.96,
    "black_is_deep": True,
    "flatten_background": False,
    "background_threshold": 0.88,
    "background_value": 1.0,
    "smooth": "bilateral",
    "smooth_diameter": 9,
    "smooth_strength": 0.08,
    "sharpen": 0.2,
    "sharpen_sigma": 2.0,
    # Stage A — input conditioning toggles (defaults are no-ops).
    "input_white_balance": False,
    "input_clahe": False,
    "input_clahe_clip": 2.0,
    "input_clahe_grid": 8,
    "input_denoise": False,
    "input_denoise_strength": 5.0,
    "input_remove_specular": False,
    "input_specular_threshold": 245,
    "input_max_dim": 0,
    # Stage C extras.
    "edge_refine": False,
    "edge_refine_diameter": 9,
    "edge_refine_sigma_color": 0.08,
    "edge_refine_sigma_space": 6.0,
    "dither": False,
    "dither_levels": 256,
    # Calibration / pass-aware preview.
    "target_depth_um": 0.0,        # 0 = use LUT.max_depth_um (or LUT off)
    "posterize_passes": 0,         # 0 = off
    # Stage B — photo-detail injection. Lowered to 0.10 to match autofit and
    # the ReliefGenerater reference (α≈0.05, plus a touch since we high-pass
    # rather than blend luminance directly). See IMPLEMENTATION_PLAN.md §2.
    "detail_mode": "off",          # "off" | "luminance" | "highpass" | "both"
    "detail_strength": 0.10,
    "detail_highpass_radius": 9,
    "detail_subject_mask": True,
    "detail_invert": False,
    # Phase 2 — subject isolation: hard-flatten background to a known plane.
    # Off by default so existing tests/profiles keep their current behaviour;
    # the sculptok-portrait preset and the SPA Wizard turn it on.
    "subject_mask_enabled": False,
    "subject_mask_backend": "rembg",
    "subject_mask_feather_px": 3,
    "subject_mask_threshold": 0.5,
    # Phase 3 — relief composite: blend bulk depth with FC-integrated normals.
    # `relief_strength` is w_micro in the convex combination (w_bulk = 1-w).
    "relief_enabled": False,
    "relief_strength": 0.3,
    "relief_normals_backend": "finite_diff",
    "relief_pad_fraction": 0.25,
    # Phase 3b — gradient-domain depth compression (Kerber): shallow-but-sharp
    # relief by amplifying small gradients and attenuating depth-jumps.
    "depth_unsharp_enabled": False,
    "depth_unsharp_gamma": 0.7,
    "depth_unsharp_blend": 0.5,
    # Phase 4 — face-aware per-region depth weighting. The single biggest
    # quality lever for portraits. No-op when no face is detected.
    "face_relief_enabled": False,
    "face_relief_strength": 1.0,
    # Auto-orient: rotate the input so the inter-pupillary line is level,
    # using the same MediaPipe face mesh as face_relief. No-op without a face.
    "auto_orient_face": False,
    # Delighting (Marigold-IID-Appearance): replace the photo with its
    # albedo before depth/normals so specular highlights and cast shadows
    # don't read as concavities/bumps. Opt-in (CC-BY-NC-4.0).
    "delight_enabled": False,
    "delight_backend": "marigold_iid",
    # Photo-guided bilateral cross-filter on the *raw depth* (pre-
    # normalisation). Sharpens depth edges to photo edges on hair / fabric
    # silhouettes without compressing the dynamic range. Cheap when cv2
    # was built with ximgproc; no-op otherwise.
    "depth_bilateral_enabled": False,
    "depth_bilateral_diameter": 9,
    "depth_bilateral_sigma_color": 0.05,
    "depth_bilateral_sigma_space": 8.0,
    # Pre-depth super-resolution. Upscale sub-threshold inputs so the depth
    # model sees more pixels — this is the single largest contributor to
    # micro-relief sharpness when the source photo is small (the case for
    # most reference photos pulled from the web). ``lanczos`` is free and
    # always works; ``realesrgan-x4plus`` (BSD-3) gives a real quality lift
    # at the cost of ~17 MB weights + a few seconds of GPU time.
    "pre_upscale_enabled": False,
    "pre_upscale_resolver": "lanczos",
    "pre_upscale_target_long_side": 1024,
    # External heightmap input: when set, the depth network is bypassed
    # and the supplied PNG/TIFF is used as the depth source. Use case:
    # operator brings a sculptok / meshy.ai relief render alongside the
    # original photo and our pipeline provides the engraving toolchain
    # (subject mask, color quantisation, multi-pass .lbrn2/.clb export,
    # calibration, burn-time, signature) on top.
    "external_heightmap_path": "",
    "external_heightmap_polarity": "bright_raised",   # | dark_raised | auto
    "external_heightmap_auto_stretch": True,
    "external_heightmap_use_subject_mask": True,
    "external_heightmap_resampler": "realesrgan-x4plus",
    # Relief stylization: pass the photo + DAv2 depth through ControlNet-
    # Depth + bas-relief prompt to produce a stylized relief render, then
    # re-estimate depth on that stylized output. This recovers the high-
    # frequency surface texture (fur strands, beard hair, fabric weave,
    # sculpted facial planes) that monocular depth on natural photos
    # can't see — the architectural leap from "depth pipeline" to
    # "stylize-then-depth" pipeline. Heavy: ~12 GB weights downloaded on
    # first use, ~30-60 s/render on GPU. Opt-in.
    "relief_stylize_enabled": False,
    "relief_stylize_backend": "sdxl_controlnet_depth",
    "relief_stylize_steps": 25,
    "relief_stylize_guidance": 7.5,
    "relief_stylize_controlnet_strength": 0.95,
    "relief_stylize_seed": 0,            # 0 = random, otherwise reproducible
    "relief_stylize_blend": 1.0,         # 0 = original depth, 1 = stylized depth
    # Photo-tonal pass — low-power dithered photographic luminance overlay
    # layered on top of the carved relief. Distinct from the heightmap-
    # derived SHADING pass: that one comes from the depth field's mid-
    # frequency band; this one comes from the actual photo's grayscale
    # so it captures skin tone, fabric pattern, hair shading the depth
    # network can't see. Off by default; flip on for portrait jobs where
    # you want a photographic hint underneath the sculpted relief.
    "photo_tonal_enabled": False,
    "photo_tonal_invert": False,
    "photo_tonal_dither": True,
    "photo_tonal_levels": 32,
    "photo_tonal_strength": 0.7,
    # How deep the photo-tonal layer can carve, as a fraction of the
    # full engraving budget. 0.4 = max 40 % depth — tonal contrast on
    # top of the deeper relief without competing with the form pass.
    "photo_tonal_depth_fraction": 0.4,
    # Signature pass — render a small text label into the configured corner.
    # An empty string disables the signature pass entirely.
    "signature_text": "",
    "signature_corner": "br",
    "signature_height_fraction": 0.04,
    "signature_margin_fraction": 0.03,
    # How deep the signature carves into the surface, as a fraction of the
    # heightmap range (0 = surface, 1 = deepest engraving). 0.6 is a
    # legible mark on most materials without competing with the form.
    "signature_depth_fraction": 0.6,
}

_HEIGHTMAP_KEYS = set(DEFAULT_SETTINGS.keys())
_INPUT_PREFIX = "input_"


@dataclass
class InferenceConfig:
    model_name: str = "ZoeD_NK"
    device: str | None = None       # None -> resolve from app settings
    pad_input: bool = True
    with_flip_aug: bool = True
    tile_size: int = 0
    tile_overlap: int = 128
    precision: str | None = None    # None -> resolve from app settings ("auto"/"fp32"/"fp16"/"bf16")
    inference_resolution: int = 0   # 0 = full; otherwise downscale longest side fed to ZoeDepth


@dataclass
class PreviewResult:
    heightmap: np.ndarray            # float32 in [0, 1]
    preview_image: Image.Image       # shaded relief PNG-ready
    settings: Dict[str, Any]
    elapsed_s: float
    image_hash: str


@dataclass
class ExportRequest:
    output_dir: Path
    base_stem: str
    write_preview: bool = True
    write_calibration_ramp: bool = False
    naming: str = "overwrite"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    keep_history: bool = False
    # Pass-stack / LightBurn-project export. Off by default so existing
    # callers (test fixtures, headless integrations) keep their current
    # bundle shape; opt in with ``write_lbrn2=True`` to get a one-shot
    # drop-into-LightBurn artefact.
    write_lbrn2: bool = False
    write_pass_pngs: bool = False
    write_clb: bool = False                  # standalone Cut Library export
    lightburn_card: str | None = None       # None = profile_name or DEFAULT card
    n_color_passes: int = 0                  # 0 = skip color quantisation
    pass_toggles: Dict[str, bool] = field(default_factory=dict)
    # Physical print size for burn-time estimation. Defaults to a square
    # 50 mm at the heightmap's pixel aspect ratio if not provided.
    print_width_mm: float | None = None
    print_height_mm: float | None = None


@dataclass
class ExportBundle:
    lightburn_png: Path
    master16_png: Path
    preview_png: Path | None
    ramp_png: Path | None
    settings_json: Path
    elapsed_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Pass-stack outputs (populated only when ``write_lbrn2`` /
    # ``write_pass_pngs`` were requested).
    lbrn2_path: Path | None = None
    clb_path: Path | None = None
    pass_png_paths: Dict[str, Path] = field(default_factory=dict)
    # QA findings (heuristic warnings). Always populated; an empty list
    # means everything looks fine.
    qa_findings: list[Dict[str, str]] = field(default_factory=list)
    # Burn-time estimate, populated when ``write_lbrn2`` is set so the
    # caller has the same information the LightBurn user will see.
    burn_estimate: Dict[str, Any] | None = None


def merge_profile_settings(
    profile_data: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Defaults <- profile heightmap block <- explicit overrides."""
    settings = dict(DEFAULT_SETTINGS)
    if profile_data:
        for key, value in (profile_data.get("heightmap") or {}).items():
            settings[key] = value
        if "black_is_deep" in profile_data:
            settings["black_is_deep"] = bool(profile_data["black_is_deep"])
        # Top-level profile blocks consumed by Stage C downstream.
        if "calibration_lut" in profile_data:
            settings["calibration_lut"] = profile_data["calibration_lut"]
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key in _HEIGHTMAP_KEYS or key in {"calibration_lut", "target_depth_um"}:
                settings[key] = value
    return settings


def _hash_pil(image: Image.Image) -> str:
    return _exporter.hash_image(image)


def _depth_cache_key(image_hash: str, cfg: InferenceConfig, resolved_device: str, resolved_precision: str) -> str:
    payload = (
        f"{image_hash}|{cfg.model_name}|{cfg.pad_input}|{cfg.with_flip_aug}|"
        f"{cfg.tile_size}|{cfg.tile_overlap}|{cfg.inference_resolution}|"
        f"{resolved_device}|{resolved_precision}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _autocast_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None  # fp32 -> no autocast


class _NullCtx:
    def __enter__(self):  # noqa: D401
        return None
    def __exit__(self, *exc):
        return False


def _resize_depth(depth: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Bilinearly resize a 2-D float depth map to (target_w, target_h)."""
    try:
        import cv2
        return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # PIL fallback (slower, but always available).
        img = Image.fromarray(depth, mode="F")
        return np.asarray(img.resize((target_w, target_h), Image.BILINEAR), dtype=np.float32)


class HeightmapService:
    """Stateful service that owns the loaded model and a small depth cache.

    Construct once per process. Reuses the model across calls; caches the most
    recent depth map keyed by (input hash, inference config) so slider tweaks
    don't trigger re-inference.
    """

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        model_loader: Callable[[str, str], tuple[Any, str]] | None = None,
    ) -> None:
        self.app_settings = app_settings or AppSettings()
        self._model_loader = model_loader or _default_model_loader
        self._model = None
        self._loaded_model_name: str | None = None
        self._loaded_device: str | None = None
        self._depth_cache: Dict[str, np.ndarray] = {}
        self._depth_cache_order: list[str] = []
        self._cache_capacity = 4
        # Subject-mask alpha cache, keyed by (image_hash, backend). Mask
        # inference is the slowest opt-in stage; tweaking heightmap sliders
        # must not re-invoke rembg/BiRefNet every render.
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._mask_cache_order: list[str] = []
        # Lazy-loaded subject-masker instances, keyed by backend name.
        self._maskers: Dict[str, Any] = {}

    # ------------------------------------------------------------------ models
    def ensure_model(self, model_name: str, device: str | None = None):
        target_device = device or resolve_device(self.app_settings.inference.device)
        if (
            self._model is not None
            and self._loaded_model_name == model_name
            and self._loaded_device == target_device
        ):
            return self._model, target_device
        self._model, self._loaded_device = self._model_loader(model_name, target_device)
        self._loaded_model_name = model_name
        # Changing the model invalidates depth cache.
        self._depth_cache.clear()
        self._depth_cache_order.clear()
        return self._model, self._loaded_device

    def unload_model(self) -> None:
        self._model = None
        self._loaded_model_name = None
        self._loaded_device = None
        self._depth_cache.clear()
        self._depth_cache_order.clear()
        self._mask_cache.clear()
        self._mask_cache_order.clear()
        self._maskers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------- depth
    def infer_depth(self, image: Image.Image, cfg: InferenceConfig) -> tuple[np.ndarray, str]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image_hash = _hash_pil(image)

        target_device = cfg.device or resolve_device(self.app_settings.inference.device)
        precision = resolve_precision(
            cfg.precision if cfg.precision is not None else self.app_settings.inference.precision,
            target_device,
        )
        cache_key = _depth_cache_key(image_hash, cfg, target_device, precision)
        cached = self._depth_cache.get(cache_key)
        if cached is not None:
            return cached, image_hash

        model, _ = self.ensure_model(cfg.model_name, target_device)

        # Optional downscale before inference (huge speedup, modest quality cost).
        original_w, original_h = image.size
        infer_image = image
        if cfg.inference_resolution and cfg.inference_resolution > 0:
            longest = max(original_w, original_h)
            if longest > cfg.inference_resolution:
                scale = cfg.inference_resolution / float(longest)
                new_w = max(1, int(round(original_w * scale)))
                new_h = max(1, int(round(original_h * scale)))
                infer_image = image.resize((new_w, new_h), Image.LANCZOS)

        autocast_dtype = _autocast_dtype(precision)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and target_device.startswith("cuda")
            else _NullCtx()
        )

        with autocast_ctx:
            if cfg.tile_size and cfg.tile_size > 0:
                depth = infer_tiled_pil(
                    model,
                    infer_image,
                    tile_size=cfg.tile_size,
                    overlap=cfg.tile_overlap,
                    pad_input=cfg.pad_input,
                    with_flip_aug=cfg.with_flip_aug,
                )
            else:
                depth = model.infer_pil(
                    infer_image,
                    pad_input=cfg.pad_input,
                    with_flip_aug=cfg.with_flip_aug,
                )

        depth = np.asarray(depth, dtype=np.float32)

        # Restore depth to the conditioned image's full size so downstream
        # stages (guided refine, dither, exporter) operate at native res.
        if depth.shape[1] != original_w or depth.shape[0] != original_h:
            depth = _resize_depth(depth, original_w, original_h)

        self._cache_depth(cache_key, depth)
        return depth, image_hash

    def _cache_depth(self, key: str, depth: np.ndarray) -> None:
        if key in self._depth_cache:
            return
        self._depth_cache[key] = depth
        self._depth_cache_order.append(key)
        while len(self._depth_cache_order) > self._cache_capacity:
            evict = self._depth_cache_order.pop(0)
            self._depth_cache.pop(evict, None)

    # ---------------------------------------------------------------- preview
    def render(
        self,
        image: Image.Image,
        settings: Mapping[str, Any],
        cfg: InferenceConfig,
    ) -> PreviewResult:
        start = time.perf_counter()
        # Stage A — input conditioning (EXIF, WB, CLAHE, denoise, specular).
        cond_payload = {
            k[len(_INPUT_PREFIX):]: v for k, v in settings.items() if k.startswith(_INPUT_PREFIX)
        }
        cond_payload.setdefault("auto_orient", True)
        if "max_dim" in cond_payload and cond_payload["max_dim"] is not None:
            cond_payload["max_input_dim"] = cond_payload.pop("max_dim")
        conditioned = condition_input(image, settings_from_mapping(cond_payload))

        # Pre-depth super-resolution (opt-in). Upscale small inputs so the
        # depth backbone sees more pixels. Real-ESRGAN x4 (BSD-3) is the
        # quality option; ``lanczos`` is the zero-dep fallback.
        if bool(settings.get("pre_upscale_enabled", False)):
            from .super_resolution import auto_upscale
            try:
                conditioned = auto_upscale(
                    conditioned,
                    target_long_side=int(settings.get("pre_upscale_target_long_side", 1024)),
                    resolver_key=str(settings.get("pre_upscale_resolver", "lanczos")),
                    device=self._loaded_device or "cpu",
                )
            except Exception:
                # Never fail a render because the SR backend can't load —
                # the user explicitly opted in but the resolver isn't ready.
                pass

        # Face-aware auto-orient (opt-in). Runs after EXIF/conditioning so
        # we don't double-rotate. No-op when no face is detected.
        if bool(settings.get("auto_orient_face", False)):
            from .face_relief import auto_orient_to_face
            conditioned = auto_orient_to_face(conditioned)

        # Delight before depth inference so specular / shadow cues stop
        # leaking into geometry. Skipped silently when the backend fails
        # to load (missing diffusers, missing weights, etc.).
        if bool(settings.get("delight_enabled", False)):
            conditioned = self._maybe_delight(conditioned, settings)

        # External heightmap branch: skip our own depth network and use a
        # precomputed heightmap from sculptok / meshy / hand-authored. The
        # rest of the pipeline (mask, color quant, pass-stack, .lbrn2)
        # runs unchanged on top.
        external_path = str(settings.get("external_heightmap_path", "") or "").strip()
        if external_path:
            depth, image_hash = self._load_external_heightmap_as_depth(
                conditioned, external_path, settings,
            )
        else:
            depth, image_hash = self.infer_depth(conditioned, cfg)
        guide_rgb = np.asarray(conditioned)

        # Relief stylization (opt-in). Take the photo + our depth through
        # ControlNet-Depth + a bas-relief diffusion prompt; the diffusion
        # model hallucinates fur / beard / embroidery in the right
        # relief style; we then re-estimate depth on the stylized output
        # and blend with the original. Falls back silently if the
        # diffusers pipeline can't load.
        if bool(settings.get("relief_stylize_enabled", False)):
            try:
                depth = self._stylize_and_redepth(
                    conditioned, depth, settings, cfg,
                )
            except Exception:
                # Stylization is opt-in convenience; never break a render
                # over a missing weight or an OOM.
                pass

        # Photo-guided bilateral cross-filter on raw depth (opt-in).
        if bool(settings.get("depth_bilateral_enabled", False)):
            from .heightmap import bilateral_cross_filter_depth
            depth = bilateral_cross_filter_depth(
                depth,
                guide_rgb,
                diameter=int(settings.get("depth_bilateral_diameter", 9)),
                sigma_color=float(settings.get("depth_bilateral_sigma_color", 0.05)),
                sigma_space=float(settings.get("depth_bilateral_sigma_space", 8.0)),
            )

        # Phase 3 — bulk depth + FC micro-relief composite (opt-in).
        if bool(settings.get("relief_enabled", False)):
            depth = self._compose_relief(depth, conditioned, settings)

        # Phase 3b — gradient-domain compression (Kerber). Runs in depth
        # space, before normalisation, so the percentile clip downstream
        # operates on the compressed gradients and leaves more headroom
        # for the surface micro-relief.
        if bool(settings.get("depth_unsharp_enabled", False)):
            from .depth_unsharp import gradient_domain_compress
            depth = gradient_domain_compress(
                depth,
                gamma=float(settings.get("depth_unsharp_gamma", 0.7)),
                blend=float(settings.get("depth_unsharp_blend", 0.5)),
            )

        heightmap = process_depth_to_heightmap(depth, dict(settings), guide_rgb=guide_rgb)

        # Phase 2 — hard subject isolation (opt-in). Runs after the heightmap
        # is in [0, 1] so we can flatten the background to a known plane in
        # heightmap coordinates regardless of `black_is_deep`.
        if bool(settings.get("subject_mask_enabled", False)):
            heightmap = self._apply_subject_mask(heightmap, conditioned, image_hash, settings)

        # Phase 4 — face-aware per-region depth weighting (the moat).
        # Runs last so it operates on the final, mask-flattened heightmap.
        if bool(settings.get("face_relief_enabled", False)):
            from .face_relief import apply_face_relief
            heightmap = apply_face_relief(
                heightmap,
                conditioned,
                strength=float(settings.get("face_relief_strength", 1.0)),
                black_is_deep=bool(settings.get("black_is_deep", True)),
            )

        preview = render_preview(heightmap)
        elapsed = time.perf_counter() - start
        return PreviewResult(
            heightmap=heightmap,
            preview_image=preview,
            settings=dict(settings),
            elapsed_s=elapsed,
            image_hash=image_hash,
        )

    # ----------------------------------------------- relief & subject mask
    def _compose_relief(
        self,
        depth: np.ndarray,
        conditioned: Image.Image,
        settings: Mapping[str, Any],
    ) -> np.ndarray:
        """Add normals→FC-integrated *micro-texture* on top of the bulk depth.

        The bulk depth already carries form (silhouette + body shape). FC
        integration of low-resolution normals reproduces that same form, so a
        convex blend just compresses the depth's dynamic range. Instead we
        keep only the high-frequency content of the integrated relief and
        add it to the depth — the depth keeps its form, the FC contribution
        is purely surface texture (fabric weave, beard, fur).
        """
        from .frankot_chellappa import integrate_normals
        from .normals import depth_to_normals, load_estimator
        from .relief import normalise_unit

        backend = str(settings.get("relief_normals_backend", "finite_diff"))
        if backend == "finite_diff":
            normals = depth_to_normals(depth)
        else:
            try:
                estimator, _ = load_estimator(backend, self._loaded_device or "cpu")
                normals = estimator.infer(conditioned)
            except (KeyError, RuntimeError, ImportError):
                # Fall back silently to finite-diff so a failed estimator load
                # never breaks the render. The user sees the same composite
                # they'd get with `relief_normals_backend="finite_diff"`.
                normals = depth_to_normals(depth)

        relief = integrate_normals(
            normals,
            pad_fraction=float(settings.get("relief_pad_fraction", 0.25)),
        )
        # Convention align: FC integrates so brighter = closer-to-camera, but
        # `depth` here is "larger = farther" (ZoeDepth/DAv2 convention).
        relief = -relief

        # Percentile-clip the silhouette spikes (|Nz|→0 at occluding contours).
        lo = float(np.percentile(relief, 1.0))
        hi = float(np.percentile(relief, 99.0))
        if hi > lo:
            relief = np.clip(relief, lo, hi)

        # Keep only high-frequency content of the relief — Gaussian-blur it
        # and subtract. The cutoff scales with the longer side so the same
        # `relief_strength` knob behaves the same regardless of image size.
        sigma = max(2.0, 0.02 * max(relief.shape))
        blurred = self._gaussian_blur(relief, sigma)
        relief_hp = relief - blurred

        # Rescale the high-frequency relief into the depth's working range so
        # the additive combine has a meaningful unit. We want a 1.0-strength
        # mix to add roughly the depth's full dynamic range as texture, so
        # normalise the relief's percentile-trimmed range against the depth's.
        d_lo = float(np.percentile(depth, 1.0))
        d_hi = float(np.percentile(depth, 99.0))
        depth_span = max(1e-6, d_hi - d_lo)
        r_abs = float(np.percentile(np.abs(relief_hp), 99.0))
        if r_abs > 1e-6:
            relief_hp = relief_hp * (depth_span / r_abs)

        strength = float(np.clip(settings.get("relief_strength", 0.3), 0.0, 1.0))
        return (depth.astype(np.float32) + strength * relief_hp).astype(np.float32)

    def _load_external_heightmap_as_depth(
        self,
        conditioned: Image.Image,
        path: str,
        settings: Mapping[str, Any],
    ) -> tuple[np.ndarray, str]:
        """Load a sculptok/meshy heightmap PNG and return it shaped like a depth array.

        Returns ``(heightmap_in_zoedepth_polarity, image_hash)``. The
        returned array is in our internal "larger value = farther from
        camera" convention so :func:`process_depth_to_heightmap` runs
        unchanged.

        Internal flow:
            1. Compute the BiRefNet alpha if the user opted in (it's the
               canonical subject silhouette for the photo).
            2. Load the external PNG, resize to the photo, polarity-
               normalise, and (optionally) auto-stretch the in-subject
               range to fill the engraving budget.
            3. Convert from "bright = surface (1.0)" → "larger = farther"
               by subtracting from 1.0, so downstream ``normalize_depth``
               + ``orient_for_lightburn(black_is_deep=True)`` produces
               the same heightmap polarity LightBurn expects.
        """
        from .exporter import hash_image
        from .external_heightmap import (
            DEFAULT_AUTO_STRETCH,
            DEFAULT_POLARITY,
            DEFAULT_RESAMPLE,
            EXTERNAL_DEPTH_DEEP_LIMIT,
            EXTERNAL_DEPTH_SURFACE_LIMIT,
            fit_external_heightmap_to_photo,
        )

        photo_size = conditioned.size  # (W, H)
        image_hash = hash_image(conditioned)

        # Subject mask — only computed when the operator wants it
        # applied. We reuse the subject_mask cache so a manual mask
        # override flow stays a one-liner.
        subject_alpha: np.ndarray | None = None
        if bool(settings.get("external_heightmap_use_subject_mask", True)):
            backend = str(settings.get("subject_mask_backend", "rembg"))
            cache_key = f"{image_hash}|{backend}"
            cached = self._mask_cache.get(cache_key)
            if cached is not None:
                subject_alpha = cached
            else:
                try:
                    masker = self._get_masker(backend)
                    subject_alpha = np.asarray(
                        masker.infer(conditioned), dtype=np.float32,
                    )
                    self._cache_mask(cache_key, subject_alpha)
                except Exception:
                    subject_alpha = None

        heightmap = fit_external_heightmap_to_photo(
            path,
            photo_size=photo_size,
            subject_alpha=subject_alpha,
            polarity=str(settings.get("external_heightmap_polarity", DEFAULT_POLARITY)),
            auto_stretch=bool(settings.get("external_heightmap_auto_stretch", DEFAULT_AUTO_STRETCH)),
            deep_limit=EXTERNAL_DEPTH_DEEP_LIMIT,
            surface_limit=EXTERNAL_DEPTH_SURFACE_LIMIT,
            background_value=float(settings.get("background_value", 1.0)),
            resampler_key=str(settings.get("external_heightmap_resampler", DEFAULT_RESAMPLE)),
            device=self._loaded_device or "cpu",
        )

        # ``heightmap`` is in LightBurn convention: 1.0=surface, 0.0=deepest.
        # ``process_depth_to_heightmap`` expects ZoeDepth convention
        # (larger=farther). Invert; downstream orient flips it back.
        depth_like = (1.0 - heightmap).astype(np.float32)
        return depth_like, image_hash

    def _stylize_and_redepth(
        self,
        conditioned: Image.Image,
        depth: np.ndarray,
        settings: Mapping[str, Any],
        cfg: "InferenceConfig",
    ) -> np.ndarray:
        """Pass through ControlNet-Depth bas-relief stylizer, re-estimate depth.

        The stylizer hallucinates relief-style detail conditioned on our
        depth map; running depth inference on its output gives us a depth
        map with that detail baked in. The result is blended with the
        original via ``relief_stylize_blend``.
        """
        from .relief_stylizer import (
            DEFAULT_CONTROLNET_STRENGTH,
            DEFAULT_GUIDANCE_SCALE,
            DEFAULT_INFERENCE_STEPS,
            load_stylizer,
        )

        backend = str(settings.get("relief_stylize_backend", "sdxl_controlnet_depth"))
        device = self._loaded_device or "cpu"
        stylizer, _ = load_stylizer(backend, device)

        seed_raw = int(settings.get("relief_stylize_seed", 0) or 0)
        stylized = stylizer.stylize(
            conditioned,
            depth,
            steps=int(settings.get("relief_stylize_steps", DEFAULT_INFERENCE_STEPS)),
            guidance=float(settings.get("relief_stylize_guidance", DEFAULT_GUIDANCE_SCALE)),
            controlnet_strength=float(settings.get(
                "relief_stylize_controlnet_strength", DEFAULT_CONTROLNET_STRENGTH,
            )),
            seed=(seed_raw if seed_raw > 0 else None),
        )

        # Run a fresh depth inference on the stylized image. We bypass
        # the depth cache because the input is different from the cached
        # one; using ``infer_depth`` directly with the stylized PIL is
        # the right abstraction.
        depth_stylized, _ = self.infer_depth(stylized, cfg)

        blend = float(np.clip(settings.get("relief_stylize_blend", 1.0), 0.0, 1.0))
        if blend >= 1.0:
            return depth_stylized
        if blend <= 0.0:
            return depth
        return (1.0 - blend) * depth + blend * depth_stylized

    def _maybe_delight(self, image: Image.Image, settings: Mapping[str, Any]) -> Image.Image:
        """Run the configured delighter; on any failure, return ``image`` unchanged.

        Caching: the loaded delighter is held by the underlying module-level
        cache in :mod:`zoedepth.laser.delighting`, so repeated renders on the
        same image don't reload the diffusion weights.
        """
        from .delighting import load_delighter

        backend = str(settings.get("delight_backend", "marigold_iid"))
        try:
            delighter, _ = load_delighter(
                backend, self._loaded_device or "cpu",
            )
            return delighter.albedo(image)
        except Exception:
            return image

    @staticmethod
    def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
        """Light Gaussian blur with cv2 fallback to scipy/uniform."""
        try:
            import cv2
            ksize = max(3, int(round(sigma * 6)) | 1)
            return cv2.GaussianBlur(arr.astype(np.float32), (ksize, ksize),
                                    sigmaX=float(sigma), sigmaY=float(sigma))
        except Exception:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr.astype(np.float32), sigma=float(sigma))

    def _get_masker(self, backend: str):
        """Lazy-load and cache a subject masker by backend key."""
        from .subject_mask import load_masker

        cached = self._maskers.get(backend)
        if cached is not None:
            return cached
        device = self._loaded_device or resolve_device(self.app_settings.inference.device)
        instance, _ = load_masker(backend, device)
        self._maskers[backend] = instance
        return instance

    def _cache_mask(self, key: str, alpha: np.ndarray) -> None:
        if key in self._mask_cache:
            return
        self._mask_cache[key] = alpha
        self._mask_cache_order.append(key)
        while len(self._mask_cache_order) > self._cache_capacity:
            evict = self._mask_cache_order.pop(0)
            self._mask_cache.pop(evict, None)

    def _apply_subject_mask(
        self,
        heightmap: np.ndarray,
        conditioned: Image.Image,
        image_hash: str,
        settings: Mapping[str, Any],
    ) -> np.ndarray:
        """Run the configured subject masker and flatten the background plane."""
        from .subject_mask import (
            DEFAULT_BACKGROUND_PLANE,
            DEFAULT_BINARY_THRESHOLD,
            DEFAULT_FEATHER_PX,
            compose_mask_with_heightmap,
        )

        backend = str(settings.get("subject_mask_backend", "rembg"))
        cache_key = f"{image_hash}|{backend}"
        alpha = self._mask_cache.get(cache_key)
        if alpha is None:
            try:
                masker = self._get_masker(backend)
                alpha = np.asarray(masker.infer(conditioned), dtype=np.float32)
            except Exception:
                # If the chosen backend can't be loaded (offline, missing weight,
                # GPU OOM), skip masking rather than failing the render.
                return heightmap
            if alpha.shape != heightmap.shape:
                # PIL ordering: (W, H); numpy ordering: (H, W).
                alpha_pil = Image.fromarray(
                    np.clip(alpha, 0.0, 1.0) * 255.0
                ).convert("L").resize(
                    (heightmap.shape[1], heightmap.shape[0]),
                    Image.BILINEAR,
                )
                alpha = np.asarray(alpha_pil, dtype=np.float32) / 255.0
            self._cache_mask(cache_key, alpha)

        return compose_mask_with_heightmap(
            heightmap.astype(np.float32, copy=False),
            alpha,
            background_value=float(settings.get("background_value", DEFAULT_BACKGROUND_PLANE)),
            binary_threshold=float(settings.get("subject_mask_threshold", DEFAULT_BINARY_THRESHOLD)),
            feather_px=int(settings.get("subject_mask_feather_px", DEFAULT_FEATHER_PX)),
        )

    # ---------------------------------------------------------- auto-fit
    def analyze_for_autofit(
        self,
        image: Image.Image,
        settings: Mapping[str, Any],
        cfg: InferenceConfig,
    ) -> Dict[str, Any]:
        """Run conditioning + (cached) depth, return suggested override values.

        The returned dict uses the same slider keys (``near_percentile``,
        ``far_percentile``, ``gamma``, ``deep_limit``, ``surface_limit``).
        """
        from .autofit import autofit_overrides_from_depth

        cond_payload = {
            k[len(_INPUT_PREFIX):]: v for k, v in settings.items() if k.startswith(_INPUT_PREFIX)
        }
        cond_payload.setdefault("auto_orient", True)
        if "max_dim" in cond_payload and cond_payload["max_dim"] is not None:
            cond_payload["max_input_dim"] = cond_payload.pop("max_dim")
        conditioned = condition_input(image, settings_from_mapping(cond_payload))
        depth, _ = self.infer_depth(conditioned, cfg)
        return autofit_overrides_from_depth(depth)

    # ----------------------------------------------------------------- export
    def export(
        self,
        image: Image.Image,
        settings: Mapping[str, Any],
        cfg: InferenceConfig,
        request: ExportRequest,
        *,
        profile_name: str | None = None,
        profile_data: Mapping[str, Any] | None = None,
        input_path: str | Path | None = None,
    ) -> ExportBundle:
        start = time.perf_counter()
        result = self.render(image, settings, cfg)

        request.output_dir.mkdir(parents=True, exist_ok=True)
        stem = _exporter.resolve_export_stem(
            request.output_dir,
            request.base_stem,
            naming=request.naming,
            timestamp_format=request.timestamp_format,
            keep_history=request.keep_history,
        )

        # New layout (May 2026): each export gets its own directory.
        #
        #   <output_dir>/<stem>/
        #     final/         <- drag this folder into LightBurn
        #       project.lbrn2 + cut-library.clb + lightburn.png
        #       master16.png + pass_*.png (siblings of .lbrn2 so its
        #       relative SourceFile refs continue to resolve)
        #     work/          <- support / debug artefacts
        #       preview.png + settings.json + ramp.png + sculptok PNG
        #
        # Splitting "deliverables" from "work" makes it obvious what to
        # transfer to a laser-shop machine and what stays on the design
        # workstation.
        bundle_root = request.output_dir / stem
        final_dir = bundle_root / "final"
        work_dir = bundle_root / "work"
        final_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        lightburn_path = final_dir / "lightburn.png"
        master16_path = final_dir / "master16.png"
        preview_path = work_dir / "preview.png" if request.write_preview else None
        ramp_path = work_dir / "ramp.png" if request.write_calibration_ramp else None
        settings_path = work_dir / "settings.json"

        _exporter.save_lightburn_png(result.heightmap, lightburn_path)
        _exporter.save_master16_png(result.heightmap, master16_path)
        if preview_path is not None:
            _exporter.save_preview_png(result.preview_image, preview_path)
        if ramp_path is not None:
            _exporter.save_ramp_png(create_calibration_ramp(), ramp_path)

        pass_png_paths: Dict[str, Path] = {}
        lbrn2_path: Path | None = None
        clb_path: Path | None = None
        burn_estimate: Dict[str, Any] | None = None
        if request.write_lbrn2 or request.write_pass_pngs or request.write_clb:
            try:
                pass_png_paths, lbrn2_path, clb_path, burn_estimate = self._emit_pass_stack(
                    image,
                    heightmap=result.heightmap,
                    settings=result.settings,
                    final_dir=final_dir,
                    request=request,
                    profile_name=profile_name,
                    profile_data=profile_data,
                )
            except Exception:
                # Pass-stack emission is opt-in convenience; never break
                # the canonical PNG/master16 export over a planner hiccup.
                pass_png_paths = {}
                lbrn2_path = None
                clb_path = None
                burn_estimate = None

        # QA report runs on every export — cheap heuristics that surface
        # likely problems before the user starts a long laser job.
        from .qa import qa_report
        qa_findings = [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in qa_report(
                result.heightmap,
                photo=np.asarray(image.convert("RGB")),
                background_value=float(result.settings.get("background_value", 1.0)),
            )
        ]

        device = self._loaded_device or resolve_device(self.app_settings.inference.device)
        precision_resolved = resolve_precision(
            cfg.precision if cfg.precision is not None else self.app_settings.inference.precision,
            device,
        )
        exports_meta = {
            "lightburn_png": lightburn_path.name,
            "master16_png": master16_path.name,
            "preview_png": preview_path.name if preview_path else None,
            "ramp_png": ramp_path.name if ramp_path else None,
            "naming": request.naming,
        }
        inference_meta = {
            "model": cfg.model_name,
            "device": device,
            "pad_input": cfg.pad_input,
            "with_flip_aug": cfg.with_flip_aug,
            "tile_size": cfg.tile_size,
            "tile_overlap": cfg.tile_overlap,
            "precision": precision_resolved,
            "inference_resolution": cfg.inference_resolution,
        }
        _exporter.write_settings_json(
            settings_path,
            input_path=input_path,
            image_hash=result.image_hash,
            device=device,
            model=cfg.model_name,
            profile_name=profile_name,
            profile_data=dict(profile_data or {}),
            settings=result.settings,
            inference=inference_meta,
            exports=exports_meta,
            elapsed_s=time.perf_counter() - start,
        )

        elapsed = time.perf_counter() - start
        try:
            append_history(make_entry(
                image_hash=result.image_hash,
                settings=result.settings,
                inference=inference_meta,
                output_dir=request.output_dir,
                stem=stem,
                elapsed_s=elapsed,
                input_path=input_path,
                profile=profile_name,
                model=cfg.model_name,
                device=device,
            ))
        except OSError:
            # History is best-effort; never fail an export over it.
            pass

        return ExportBundle(
            lightburn_png=lightburn_path,
            master16_png=master16_path,
            preview_png=preview_path,
            ramp_png=ramp_path,
            settings_json=settings_path,
            elapsed_s=elapsed,
            metadata={"stem": stem, "image_hash": result.image_hash},
            lbrn2_path=lbrn2_path,
            clb_path=clb_path,
            pass_png_paths=pass_png_paths,
            qa_findings=qa_findings,
            burn_estimate=burn_estimate,
        )

    # ------------------------------------------------------------ pass stack
    def _emit_pass_stack(
        self,
        image: Image.Image,
        *,
        heightmap: np.ndarray,
        settings: Mapping[str, Any],
        final_dir: Path,
        request: ExportRequest,
        profile_name: str | None,
        profile_data: Mapping[str, Any] | None,
    ) -> tuple[Dict[str, Path], Path | None, Path | None, Dict[str, Any] | None]:
        """Build per-pass masks + PNGs, plan passes, emit .lbrn2 and .clb.

        Writes everything (.lbrn2, .clb, per-pass PNGs) into ``final_dir``
        with canonical filenames so the .lbrn2's relative ``SourceFile``
        attributes resolve to the sibling PNGs.

        Returns ``(pass_png_paths, lbrn2_path, clb_path, burn_estimate_dict)``.
        Any of them may be empty/None depending on what was requested.
        """
        from .burn_time import estimate_burn_time, format_seconds
        from .color_quantize import color_masks_for_planner, quantize_to_color_masks
        from .lbrn_writer import write_clb, write_lbrn
        from .lightburn_cards import DEFAULT_CARDS_DIR, DEFAULT_PROFILE_NAME, load_lightburn_card
        from .pass_masks import derive_pass_masks
        from .signature import render_text_signature_mask
        from .stages import (
            PASS_KIND_FORM,
            PASS_KIND_PHOTO_TONAL,
            PASS_KIND_SIGNATURE,
            plan_passes,
        )

        # 1. Resolve which LightBurn card supplies the cut settings.
        card_name = request.lightburn_card or profile_name or DEFAULT_PROFILE_NAME
        # Strip the .lbrn2 suffix if present so the lookup is uniform.
        card_stem = card_name[:-6] if card_name.endswith(".lbrn2") else card_name
        card_path = DEFAULT_CARDS_DIR / f"{card_stem}.lbrn2"
        if not card_path.exists():
            # Fall back to the default profile if the named card is missing.
            card_path = DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2"
        material = load_lightburn_card(card_path)

        # 2. Derive per-pass raster masks from the heightmap.
        kind_masks = derive_pass_masks(heightmap)

        # 2a. Optional photo-tonal mask: low-power dithered photo-luminance
        # overlay layered ON TOP of the carved relief. Distinct from the
        # heightmap-band SHADING pass — captures photographic detail
        # (skin tone, fabric pattern, hair shading) that the depth model
        # never saw. Subject-mask gated so we don't engrave photo onto
        # the flat background.
        if bool(settings.get("photo_tonal_enabled", False)):
            from .pass_masks import (
                DEFAULT_PHOTO_TONAL_LEVELS,
                DEFAULT_PHOTO_TONAL_STRENGTH,
                photo_tonal_mask,
            )

            photo_arr = np.asarray(image.convert("RGB"))
            # Resize photo to heightmap shape if they diverged (e.g.
            # because external_heightmap upscaled the heightmap).
            if photo_arr.shape[:2] != heightmap.shape:
                from PIL import Image as _PILImage
                photo_arr = np.asarray(
                    image.convert("RGB").resize(
                        (heightmap.shape[1], heightmap.shape[0]),
                        _PILImage.LANCZOS,
                    )
                )
            subject_alpha = kind_masks.get(PASS_KIND_FORM)
            kind_masks[PASS_KIND_PHOTO_TONAL] = photo_tonal_mask(
                photo_arr,
                subject_alpha,
                invert=bool(settings.get("photo_tonal_invert", False)),
                dither=bool(settings.get("photo_tonal_dither", True)),
                dither_levels=int(settings.get("photo_tonal_levels", DEFAULT_PHOTO_TONAL_LEVELS)),
                strength=float(settings.get("photo_tonal_strength", DEFAULT_PHOTO_TONAL_STRENGTH)),
            )

        # 2b. Optional signature mask: replace the default full-frame mask
        # for the signature pass with a small text rendering. Empty text
        # leaves the default in place; the user can also disable the
        # signature pass entirely via ``pass_toggles``.
        sig_text = str(settings.get("signature_text", "") or "").strip()
        if sig_text:
            kind_masks[PASS_KIND_SIGNATURE] = render_text_signature_mask(
                shape=heightmap.shape,
                text=sig_text,
                corner=str(settings.get("signature_corner", "br")),
                height_fraction=float(settings.get("signature_height_fraction", 0.04)),
                margin_fraction=float(settings.get("signature_margin_fraction", 0.03)),
            )

        # 3. Optional color quantisation against the photo.
        color_masks: Dict[str, np.ndarray] = {}
        if int(request.n_color_passes) >= 2:
            subj_mask = kind_masks.get(PASS_KIND_FORM)
            clusters = quantize_to_color_masks(
                image, k=int(request.n_color_passes), subject_mask=subj_mask,
            )
            color_masks = color_masks_for_planner(clusters)

        # 4. Plan the engraving stack.
        plan = plan_passes(
            heightmap=heightmap,
            profile=material,
            user_toggles=dict(request.pass_toggles or {}),
            masks=kind_masks,
            mask_per_color=color_masks,
        )

        # 5. Write per-pass PNGs. Each pass's layer carves only where its
        # mask says to. Two semantics:
        #   * Signature pass: depth is INDEPENDENT of the master heightmap
        #     — the laser carves the text at ``signature_depth_fraction``
        #     wherever the text mask is set (so the corner sigil is
        #     visible even on flat background).
        #   * Every other pass: engraves the master heightmap, masked to
        #     the pass's region. Surface (1.0) elsewhere so LightBurn's
        #     3D-Sliced reads "no engraving".
        sig_depth = float(np.clip(
            settings.get("signature_depth_fraction", 0.6), 0.0, 1.0,
        ))
        photo_tonal_depth = float(np.clip(
            settings.get("photo_tonal_depth_fraction", 0.4), 0.0, 1.0,
        ))
        pngs: Dict[str, Path] = {}
        for idx, ep in enumerate(plan.passes):
            # Canonical per-pass filename inside the bundle: ``pass_NN_kind.png``.
            # No stem prefix because every file in ``final_dir`` already
            # belongs to this single export.
            png_name = f"pass_{idx:02d}_{ep.kind.replace(':', '_')}.png"
            png_path = final_dir / png_name
            mask = ep.mask.astype(np.float32, copy=False)
            if ep.kind == PASS_KIND_SIGNATURE and sig_depth > 0.0:
                # Signature: fixed depth, independent of master heightmap.
                layer = 1.0 - mask * sig_depth
            elif ep.kind == PASS_KIND_PHOTO_TONAL and photo_tonal_depth > 0.0:
                # Photo tonal: depth driven by the photo's luminance,
                # NOT the master heightmap. Mask carries "fire more"
                # intensity from the dithered (1 - photo_luma) signal,
                # gated by subject silhouette in pass_masks.
                layer = 1.0 - mask * photo_tonal_depth
            else:
                layer = 1.0 - (1.0 - heightmap) * mask
            _exporter.save_master16_png(
                np.clip(layer, 0.0, 1.0).astype(np.float32),
                png_path,
            )
            pngs[ep.id] = png_path

        lbrn2_path: Path | None = None
        if request.write_lbrn2:
            lbrn2_path = final_dir / "project.lbrn2"
            write_lbrn(
                lbrn2_path,
                plan,
                pass_pngs=pngs if request.write_pass_pngs else None,
                app_version=material.app_version,
            )

        clb_path: Path | None = None
        if request.write_clb:
            clb_path = final_dir / "cut-library.clb"
            # Cut Library carries every entry the plan touched.
            entries = list({ep.cut_setting.index: ep.cut_setting for ep in plan.passes}.values())
            write_clb(clb_path, entries)

        # 6. Burn-time estimate. Default print size: 50 mm on the longest
        # heightmap dimension, preserving aspect ratio.
        h, w = heightmap.shape
        if request.print_width_mm and request.print_height_mm:
            width_mm = float(request.print_width_mm)
            height_mm = float(request.print_height_mm)
        else:
            longest = max(h, w)
            scale = 50.0 / float(longest)
            width_mm = float(request.print_width_mm or w * scale)
            height_mm = float(request.print_height_mm or h * scale)
        estimate = estimate_burn_time(plan, width_mm=width_mm, height_mm=height_mm)
        burn_estimate = {
            "width_mm": estimate.width_mm,
            "height_mm": estimate.height_mm,
            "total_seconds": estimate.total_seconds,
            "total_pretty": format_seconds(estimate.total_seconds),
            "passes": [
                {
                    "id": p.pass_id,
                    "kind": p.kind,
                    "name": p.name,
                    "seconds": p.seconds,
                    "active_fraction": p.active_fraction,
                    "pass_count": p.pass_count,
                    "pretty": format_seconds(p.seconds),
                }
                for p in estimate.passes
            ],
        }

        # If the caller didn't ask for the per-pass PNGs explicitly, drop
        # them — they were only generated to embed in the .lbrn2.
        if not request.write_pass_pngs:
            for path in pngs.values():
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            pngs = {}

        return pngs, lbrn2_path, clb_path, burn_estimate


def _default_model_loader(model_name: str, device: str) -> tuple[Any, str]:
    """Resolve ``model_name`` to a loaded depth model on ``device``.

    Lookup order: pluggable backend registry first (Depth-Anything-V2 etc.),
    then fall back to ``hubconf`` for the original ZoeDepth entrypoints.
    """
    from . import backends as _backends

    spec = _backends.get_backend(model_name)
    if spec is not None:
        return _backends.load_backend(model_name, device)

    import hubconf

    _patch_timm_beit_block_drop_path()
    if not hasattr(hubconf, model_name):
        raise ValueError(f"Unknown model entrypoint: {model_name}")
    ctor = getattr(hubconf, model_name)
    model = ctor(pretrained=True).to(device).eval()
    return model, device


_TIMM_BEIT_PATCHED = False


def _patch_timm_beit_block_drop_path() -> None:
    """Make newer timm BEiT Blocks compatible with cached MiDaS forward.

    timm >=1.0 split ``Block.drop_path`` into ``drop_path1`` (post-attention)
    and ``drop_path2`` (post-MLP). The MiDaS BEiT backbone bundled in the
    torch hub cache (intel-isl_MiDaS_master) still calls ``self.drop_path(...)``
    directly. In eval mode both drop paths are identity, so aliasing
    ``drop_path`` -> ``drop_path1`` is functionally exact for inference.
    """
    global _TIMM_BEIT_PATCHED
    if _TIMM_BEIT_PATCHED:
        return
    try:
        from timm.models import beit as _beit  # type: ignore
    except ImportError:
        _TIMM_BEIT_PATCHED = True
        return
    cls = getattr(_beit, "Block", None) or getattr(_beit, "BeitBlock", None)
    if cls is not None and not hasattr(cls, "drop_path"):
        def _drop_path_alias(self):
            return getattr(self, "drop_path1", None) or getattr(self, "drop_path2", None) or torch.nn.Identity()
        cls.drop_path = property(_drop_path_alias)
    _TIMM_BEIT_PATCHED = True


def load_profile_for_service(profile_name_or_path: str) -> Dict[str, Any]:
    """Thin wrapper so callers don't import profiles directly."""
    return load_profile(profile_name_or_path)
