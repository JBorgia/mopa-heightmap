"""HeightmapService — sculptok→LightBurn bundle orchestrator.

The heightmap source is **always** an external PNG (sculptok output, meshy
ortho-render, or hand-authored). Sculptok already produced an engraving-
ready relief; the service does NOT mutate it. We just:

    photo + heightmap PNG
        → optional pre-sculptok conditioning (CLAHE / WB / denoise / despeckle)
        → load the heightmap PNG (polarity-normalise only)
        → optional polarity invert (signet-ring / recessed mode)
        → optional subject-mask deliverable (separate artifact, not applied)
        → preview render
        → bundle: lightburn.png, master16.png, project.lbrn2,
                  cut-library.clb, per-pass PNGs, settings.json, ...

The CLI (`apps/zoe2lightburn.py`) and the API adapter own the sculptok
auto-pull: they hit sculptok, save the PNG to a temp path, and set
``settings["external_heightmap_path"]`` before calling render().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image, ImageOps

from . import exporter as _exporter
from .external_heightmap import (
    DEFAULT_POLARITY,
    load_external_heightmap,
)
from .history import append_history, make_entry
from .imgproc import condition_input, settings_from_mapping
from .preview import create_calibration_ramp, render_preview
from .profiles import load_profile
from .settings import AppSettings


DEFAULT_SETTINGS: Dict[str, Any] = {
    # Stage A — pre-sculptok input conditioning. All default-off; opt-in
    # via the wizard/profile when the source photo is dim, blurry, or has
    # heavy specular highlights. These run on the *photo* before we feed
    # it to sculptok (or before we display it next to the heightmap).
    "input_white_balance": False,
    "input_clahe": False,
    "input_clahe_clip": 2.0,
    "input_clahe_grid": 8,
    "input_denoise": False,
    "input_denoise_strength": 5.0,
    "input_remove_specular": False,
    "input_specular_threshold": 245,
    "input_max_dim": 0,

    # External heightmap source — required. The CLI/API arranges the
    # sculptok download and sets this path before calling render().
    "external_heightmap_path": "",
    "external_heightmap_polarity": "bright_raised",  # | dark_raised | auto

    # Polarity invert at write time — flips the saved heightmap so the
    # subject engraves deep instead of the background. Used for signet
    # rings and other recessed designs.
    "polarity_invert": False,

    # Subject mask deliverable. Computes a binary alpha from the photo
    # via rembg/BiRefNet and saves it as a sibling artifact. Does NOT
    # modify the heightmap — sculptok already separated subject from
    # background. The mask is consumed by the pass planner (color
    # quantisation, photo-tonal gating) and shipped as ``mask.png``.
    "subject_mask_enabled": False,
    "subject_mask_backend": "rembg",
    "subject_mask_feather_px": 3,
    "subject_mask_threshold": 0.5,

    # LightBurn 3D Sliced convention: gray=255 (white) is no engraving,
    # gray=0 (black) is the deepest cut. Sculptok output already follows
    # this. Kept as an explicit setting so callers and previews can
    # reason about polarity uniformly.
    "black_is_deep": True,
    "background_value": 1.0,

    # Heightmap dither (writes a dithered copy as the lightburn.png
    # output). Useful when collapsing 16-bit master to 8-bit with limited
    # passes; off by default.
    "dither": False,
    "dither_levels": 256,

    # Photo-tonal pass — low-power dithered photographic-luminance overlay
    # layered on top of the carved relief. Captures skin tone, fabric
    # pattern, and hair shading the heightmap doesn't carry.
    "photo_tonal_enabled": False,
    "photo_tonal_invert": False,
    "photo_tonal_dither": True,
    "photo_tonal_levels": 32,
    "photo_tonal_strength": 0.7,
    "photo_tonal_depth_fraction": 0.4,

    # Signature pass — small text rendered into one corner.
    "signature_text": "",
    "signature_corner": "br",
    "signature_height_fraction": 0.04,
    "signature_margin_fraction": 0.03,
    "signature_depth_fraction": 0.6,
}

_HEIGHTMAP_KEYS = set(DEFAULT_SETTINGS.keys())
_INPUT_PREFIX = "input_"


@dataclass
class PreviewResult:
    heightmap: np.ndarray            # float32 in [0, 1], LightBurn polarity
    preview_image: Image.Image       # shaded relief PNG-ready
    settings: Dict[str, Any]
    elapsed_s: float
    image_hash: str
    subject_alpha: np.ndarray | None = None  # (H, W) float32 in [0,1] when computed


@dataclass
class ExportRequest:
    output_dir: Path
    base_stem: str
    write_preview: bool = True
    write_calibration_ramp: bool = False
    naming: str = "overwrite"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    keep_history: bool = False
    write_lbrn2: bool = False
    write_pass_pngs: bool = False
    write_clb: bool = False
    lightburn_card: str | None = None
    n_color_passes: int = 0
    pass_toggles: Dict[str, bool] = field(default_factory=dict)
    print_width_mm: float | None = None
    print_height_mm: float | None = None
    write_subject_mask: bool = False  # ship mask.png as a sibling artifact


@dataclass
class ExportBundle:
    lightburn_png: Path
    master16_png: Path
    preview_png: Path | None
    ramp_png: Path | None
    settings_json: Path
    elapsed_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    lbrn2_path: Path | None = None
    clb_path: Path | None = None
    pass_png_paths: Dict[str, Path] = field(default_factory=dict)
    qa_findings: list[Dict[str, str]] = field(default_factory=list)
    burn_estimate: Dict[str, Any] | None = None
    subject_mask_png: Path | None = None


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
        if "calibration_lut" in profile_data:
            settings["calibration_lut"] = profile_data["calibration_lut"]
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key in _HEIGHTMAP_KEYS or key in {"calibration_lut"}:
                settings[key] = value
    return settings


def _hash_pil(image: Image.Image) -> str:
    return _exporter.hash_image(image)


class HeightmapService:
    """Stateful service: caches subject masks across renders on the same photo."""

    def __init__(self, app_settings: AppSettings | None = None) -> None:
        self.app_settings = app_settings or AppSettings()
        # Subject-mask alpha cache, keyed by (image_hash, backend). Mask
        # inference is the single slowest opt-in stage; tweaking other
        # settings must not re-invoke rembg/BiRefNet.
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._mask_cache_order: list[str] = []
        self._cache_capacity = 4
        # Lazy-loaded subject-masker instances, keyed by backend name.
        self._maskers: Dict[str, Any] = {}

    # ---------------------------------------------------------------- preview
    def render(
        self,
        image: Image.Image,
        settings: Mapping[str, Any],
    ) -> PreviewResult:
        start = time.perf_counter()

        # Stage A — pre-sculptok input conditioning (EXIF, WB, CLAHE,
        # denoise, specular). Cosmetic for the preview; the actual
        # photo-to-sculptok handoff happens in the CLI/API layer.
        cond_payload = {
            k[len(_INPUT_PREFIX):]: v
            for k, v in settings.items()
            if k.startswith(_INPUT_PREFIX)
        }
        cond_payload.setdefault("auto_orient", True)
        if "max_dim" in cond_payload and cond_payload["max_dim"] is not None:
            cond_payload["max_input_dim"] = cond_payload.pop("max_dim")
        conditioned = condition_input(image, settings_from_mapping(cond_payload))
        image_hash = _hash_pil(conditioned)

        # Heightmap source — must be supplied. Sculptok auto-pull writes
        # a temp file and sets this path; users with their own heightmap
        # set it directly.
        external_path = str(settings.get("external_heightmap_path", "") or "").strip()
        if not external_path:
            raise ValueError(
                "service.render() requires an external heightmap path. "
                "Set settings['external_heightmap_path'] to a sculptok PNG, "
                "or arrange a sculptok auto-pull at the CLI/API layer first."
            )
        if not Path(external_path).exists():
            raise FileNotFoundError(f"External heightmap not found: {external_path}")

        polarity = str(settings.get("external_heightmap_polarity", DEFAULT_POLARITY))
        # Passthrough load: bright = raised (LightBurn surface = 1.0 = no
        # engraving). No auto-stretch, no tone curve, no smoothing —
        # sculptok already shaped this for engraving.
        heightmap = load_external_heightmap(
            external_path,
            target_size=None,            # preserve sculptok's native size
            polarity=polarity,            # type: ignore[arg-type]
        )

        # Polarity invert (signet-ring mode). Flip the whole heightmap
        # so the subject engraves deep and the background stays surface.
        if bool(settings.get("polarity_invert", False)):
            heightmap = (1.0 - heightmap).astype(np.float32)

        # Subject mask — separate deliverable, NOT applied to heightmap.
        # Used by the pass planner (color quantisation, photo-tonal
        # gating) and shipped as mask.png in the bundle.
        subject_alpha: np.ndarray | None = None
        if bool(settings.get("subject_mask_enabled", False)):
            subject_alpha = self._compute_subject_mask(conditioned, image_hash, settings)

        preview = render_preview(heightmap)
        elapsed = time.perf_counter() - start
        return PreviewResult(
            heightmap=heightmap,
            preview_image=preview,
            settings=dict(settings),
            elapsed_s=elapsed,
            image_hash=image_hash,
            subject_alpha=subject_alpha,
        )

    # --------------------------------------------------------- subject mask
    def _get_masker(self, backend: str):
        from .subject_mask import load_masker

        cached = self._maskers.get(backend)
        if cached is not None:
            return cached
        instance, _ = load_masker(backend, "cpu")
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

    def _compute_subject_mask(
        self,
        conditioned: Image.Image,
        image_hash: str,
        settings: Mapping[str, Any],
    ) -> np.ndarray | None:
        backend = str(settings.get("subject_mask_backend", "rembg"))
        cache_key = f"{image_hash}|{backend}"
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            masker = self._get_masker(backend)
            alpha = np.asarray(masker.infer(conditioned), dtype=np.float32)
        except Exception:
            return None
        self._cache_mask(cache_key, alpha)
        return alpha

    # ----------------------------------------------------------------- export
    def export(
        self,
        image: Image.Image,
        settings: Mapping[str, Any],
        request: ExportRequest,
        *,
        profile_name: str | None = None,
        profile_data: Mapping[str, Any] | None = None,
        input_path: str | Path | None = None,
    ) -> ExportBundle:
        start = time.perf_counter()
        result = self.render(image, settings)

        request.output_dir.mkdir(parents=True, exist_ok=True)
        stem = _exporter.resolve_export_stem(
            request.output_dir,
            request.base_stem,
            naming=request.naming,
            timestamp_format=request.timestamp_format,
            keep_history=request.keep_history,
        )

        # Bundle layout:
        #   <output_dir>/<stem>/
        #     final/         <- drag this folder into LightBurn
        #       project.lbrn2 + cut-library.clb + lightburn.png
        #       master16.png + pass_*.png + (optional) mask.png
        #     work/          <- support / debug artefacts
        #       preview.png + settings.json + ramp.png
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

        subject_mask_path: Path | None = None
        if request.write_subject_mask and result.subject_alpha is not None:
            subject_mask_path = final_dir / "mask.png"
            mask_uint8 = np.clip(result.subject_alpha, 0.0, 1.0)
            mask_uint8 = (mask_uint8 * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(mask_uint8, mode="L").save(subject_mask_path)

        pass_png_paths: Dict[str, Path] = {}
        lbrn2_path: Path | None = None
        clb_path: Path | None = None
        burn_estimate: Dict[str, Any] | None = None
        if request.write_lbrn2 or request.write_pass_pngs or request.write_clb:
            try:
                pass_png_paths, lbrn2_path, clb_path, burn_estimate = self._emit_pass_stack(
                    image,
                    heightmap=result.heightmap,
                    subject_alpha=result.subject_alpha,
                    settings=result.settings,
                    final_dir=final_dir,
                    request=request,
                    profile_name=profile_name,
                )
            except Exception:
                pass_png_paths = {}
                lbrn2_path = None
                clb_path = None
                burn_estimate = None

        # QA report — cheap heuristics surfacing likely problems before
        # the user starts a long laser job.
        from .qa import qa_report
        qa_findings = [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in qa_report(
                result.heightmap,
                photo=np.asarray(image.convert("RGB")),
                background_value=float(result.settings.get("background_value", 1.0)),
            )
        ]

        exports_meta = {
            "lightburn_png": lightburn_path.name,
            "master16_png": master16_path.name,
            "preview_png": preview_path.name if preview_path else None,
            "ramp_png": ramp_path.name if ramp_path else None,
            "subject_mask_png": subject_mask_path.name if subject_mask_path else None,
            "naming": request.naming,
        }
        source_meta = {
            "kind": "external_heightmap",
            "path": str(result.settings.get("external_heightmap_path", "")),
            "polarity": str(result.settings.get("external_heightmap_polarity", DEFAULT_POLARITY)),
        }
        _exporter.write_settings_json(
            settings_path,
            input_path=input_path,
            image_hash=result.image_hash,
            device="cpu",
            model="external_heightmap",
            profile_name=profile_name,
            profile_data=dict(profile_data or {}),
            settings=result.settings,
            inference=source_meta,
            exports=exports_meta,
            elapsed_s=time.perf_counter() - start,
        )

        elapsed = time.perf_counter() - start
        try:
            append_history(make_entry(
                image_hash=result.image_hash,
                settings=result.settings,
                inference=source_meta,
                output_dir=request.output_dir,
                stem=stem,
                elapsed_s=elapsed,
                input_path=input_path,
                profile=profile_name,
                model="external_heightmap",
                device="cpu",
            ))
        except OSError:
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
            subject_mask_png=subject_mask_path,
        )

    # ------------------------------------------------------------ pass stack
    def _emit_pass_stack(
        self,
        image: Image.Image,
        *,
        heightmap: np.ndarray,
        subject_alpha: np.ndarray | None,
        settings: Mapping[str, Any],
        final_dir: Path,
        request: ExportRequest,
        profile_name: str | None,
    ) -> tuple[Dict[str, Path], Path | None, Path | None, Dict[str, Any] | None]:
        """Build per-pass masks + PNGs, plan passes, emit .lbrn2 and .clb."""
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
        card_stem = card_name[:-6] if card_name.endswith(".lbrn2") else card_name
        card_path = DEFAULT_CARDS_DIR / f"{card_stem}.lbrn2"
        if not card_path.exists():
            card_path = DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2"
        material = load_lightburn_card(card_path)

        # 2. Derive per-pass raster masks from the heightmap.
        kind_masks = derive_pass_masks(heightmap)

        # 2a. Optional photo-tonal mask.
        if bool(settings.get("photo_tonal_enabled", False)):
            from .pass_masks import (
                DEFAULT_PHOTO_TONAL_LEVELS,
                DEFAULT_PHOTO_TONAL_STRENGTH,
                photo_tonal_mask,
            )

            photo_arr = np.asarray(image.convert("RGB"))
            if photo_arr.shape[:2] != heightmap.shape:
                photo_arr = np.asarray(
                    image.convert("RGB").resize(
                        (heightmap.shape[1], heightmap.shape[0]),
                        Image.LANCZOS,
                    )
                )
            mask_for_photo_tonal = subject_alpha
            if mask_for_photo_tonal is None:
                mask_for_photo_tonal = kind_masks.get(PASS_KIND_FORM)
            elif mask_for_photo_tonal.shape != heightmap.shape:
                pil = Image.fromarray(
                    (np.clip(mask_for_photo_tonal, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
                    mode="L",
                ).resize(
                    (heightmap.shape[1], heightmap.shape[0]),
                    Image.BILINEAR,
                )
                mask_for_photo_tonal = np.asarray(pil, dtype=np.float32) / 255.0
            kind_masks[PASS_KIND_PHOTO_TONAL] = photo_tonal_mask(
                photo_arr,
                mask_for_photo_tonal,
                invert=bool(settings.get("photo_tonal_invert", False)),
                dither=bool(settings.get("photo_tonal_dither", True)),
                dither_levels=int(settings.get("photo_tonal_levels", DEFAULT_PHOTO_TONAL_LEVELS)),
                strength=float(settings.get("photo_tonal_strength", DEFAULT_PHOTO_TONAL_STRENGTH)),
            )

        # 2b. Optional signature mask.
        sig_text = str(settings.get("signature_text", "") or "").strip()
        if sig_text:
            kind_masks[PASS_KIND_SIGNATURE] = render_text_signature_mask(
                shape=heightmap.shape,
                text=sig_text,
                corner=str(settings.get("signature_corner", "br")),
                height_fraction=float(settings.get("signature_height_fraction", 0.04)),
                margin_fraction=float(settings.get("signature_margin_fraction", 0.03)),
            )

        # 3. Optional color quantisation against the photo (stainless
        # MOPA color anneal). Each cluster becomes its own pass with the
        # material profile's matching power/speed.
        color_masks: Dict[str, np.ndarray] = {}
        if int(request.n_color_passes) >= 2:
            subj_for_color = subject_alpha
            if subj_for_color is None:
                subj_for_color = kind_masks.get(PASS_KIND_FORM)
            clusters = quantize_to_color_masks(
                image, k=int(request.n_color_passes), subject_mask=subj_for_color,
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

        # 5. Write per-pass PNGs.
        sig_depth = float(np.clip(settings.get("signature_depth_fraction", 0.6), 0.0, 1.0))
        photo_tonal_depth = float(np.clip(
            settings.get("photo_tonal_depth_fraction", 0.4), 0.0, 1.0,
        ))
        pngs: Dict[str, Path] = {}
        for idx, ep in enumerate(plan.passes):
            png_name = f"pass_{idx:02d}_{ep.kind.replace(':', '_')}.png"
            png_path = final_dir / png_name
            mask = ep.mask.astype(np.float32, copy=False)
            if ep.kind == PASS_KIND_SIGNATURE and sig_depth > 0.0:
                layer = 1.0 - mask * sig_depth
            elif ep.kind == PASS_KIND_PHOTO_TONAL and photo_tonal_depth > 0.0:
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
            entries = list({ep.cut_setting.index: ep.cut_setting for ep in plan.passes}.values())
            write_clb(clb_path, entries)

        # 6. Burn-time estimate.
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

        if not request.write_pass_pngs:
            for path in pngs.values():
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            pngs = {}

        return pngs, lbrn2_path, clb_path, burn_estimate


def load_profile_for_service(profile_name_or_path: str) -> Dict[str, Any]:
    """Thin wrapper so callers don't import profiles directly."""
    return load_profile(profile_name_or_path)
