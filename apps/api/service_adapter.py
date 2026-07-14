"""Thin adapter: wraps mopa.* to serve the FastAPI routes.

NO business logic lives here — only the glue between HTTP schemas and the
existing Python service objects.  All math stays in mopa.*.
"""
from __future__ import annotations

import hashlib
import io
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Re-export the concrete service classes.
from mopa.service import (
    DEFAULT_SETTINGS,
    HeightmapService,
    PreviewResult,
    merge_profile_settings,
)
from mopa.settings import AppSettings, load_settings
from mopa.subject_mask import load_masker, list_maskers
from mopa.click_mask import (
    DEFAULT_CLICKER_KEY,
    DEFAULT_FLOOD_TOLERANCE,
    DEFAULT_FLOOD_MAX_FRACTION,
    POINT_LABEL_POSITIVE,
    POINT_LABEL_NEGATIVE,
    get_clicker,
    load_clicker,
    _FloodFillClicker,  # type: ignore[attr-defined]
)
from mopa.profiles import (
    list_profiles,
    load_profile,
    get_user_profiles_dir,
    validate_profile,
)
from mopa.exporter import hash_image, save_lightburn_png, save_master16_png
from mopa.heightmap import to_uint16
from mopa.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    ColorEntry,
    load_lightburn_card,
)
from mopa.lbrn_writer import build_lbrn_tree, ShapeRef
from mopa.stages import (
    plan_passes as _plan_passes,
    PASS_KIND_FIELD,
    PASS_KIND_BORDER,
    PASS_KIND_RIM,
    PASS_KIND_DEVICE,
    PASS_KIND_EXERGUE,
    PASS_KIND_FORM,
    EngravingPass,
    PassPlan,
)
from mopa.burn_time import estimate_burn_time
from mopa.zones import (
    zone_masks_from_geometry,
    zone_hm_for_pass,
    entries_from_zone_params,
    mask_from_heightmap,
    VALID_SHAPES,
)
from mopa.zone_overlays import zone_pattern_layer

from . import blob_store
from .schemas import (
    HeightmapSettings,
    MaskResponse,
    PassEntry,
    PassPlanResponse,
    RenderResponse,
    UploadResponse,
)


def _load_profile_payload(profile_name: Optional[str]) -> Dict[str, Any]:
    """Load a material profile or return an empty payload when omitted.

    Missing or invalid profiles must propagate as explicit user-input
    errors so the route layer can return 422 instead of silently falling
    back to engine defaults.
    """
    if not profile_name:
        return {}
    return load_profile(profile_name)


def _resolve_lightburn_card_path(profile_payload: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve the LightBurn color-card path for planning/export.

    Material profiles are YAML files like ``mopa_60w_brass``; they are not
    LightBurn card filenames. A profile may opt into a specific card via the
    top-level ``lightburn_card`` key. Otherwise we use the repository default
    machine card.
    """
    payload = profile_payload or {}
    card_name = payload.get("lightburn_card")
    if isinstance(card_name, str) and card_name.strip():
        card_stem = card_name.removesuffix(".lbrn2")
    else:
        card_stem = DEFAULT_PROFILE_NAME
    card_path = DEFAULT_CARDS_DIR / f"{card_stem}.lbrn2"
    if not card_path.exists():
        raise FileNotFoundError(f"LightBurn card not found: {card_path.name}")
    return card_path


def _profile_kind_color_overrides(profile_payload: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Extract per-pass LightBurn row overrides from a material profile."""
    payload = profile_payload or {}
    raw = payload.get("kind_color_overrides")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("kind_color_overrides must be a mapping of pass kind to LightBurn row name")

    overrides: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("kind_color_overrides keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("kind_color_overrides values must be non-empty strings")
        overrides[key.strip()] = value.strip()
    return overrides

# ---------------------------------------------------------------------------
# Process-singleton service
# ---------------------------------------------------------------------------

_svc_lock = threading.Lock()
_svc: Optional[HeightmapService] = None

# ---------------------------------------------------------------------------
# Masker model cache — avoids reloading ML weights on every /mask request.
# Keyed by backend name (e.g. "rembg", "birefnet", "threshold").
# ---------------------------------------------------------------------------

_masker_lock = threading.Lock()
_masker_cache: Dict[str, Any] = {}


def get_service() -> HeightmapService:
    global _svc
    with _svc_lock:
        if _svc is None:
            _svc = HeightmapService(app_settings=load_settings())
        return _svc


# ---------------------------------------------------------------------------
# Image session store (sha256 -> PIL Image in memory)
# ---------------------------------------------------------------------------

_img_lock = threading.Lock()
_images: Dict[str, Image.Image] = {}   # image_id -> PIL Image


def store_upload(image: Image.Image) -> UploadResponse:
    """Store an uploaded PIL image; return its id + metadata.

    The PNG-re-encoded bytes also go to the blob store under the same id
    (both keys are ``sha256(raw)[:40]``) so the client can fetch the source
    via ``GET /blob/{image_id}`` for the wizard preview pane.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    image_id = sha[:40]
    with _img_lock:
        if image_id not in _images:
            _images[image_id] = image.convert("RGB")
    blob_store.store_bytes(raw, "image/png")
    return UploadResponse(image_id=image_id, w=image.width, h=image.height, sha256=sha)


def get_upload(image_id: str) -> Optional[Image.Image]:
    with _img_lock:
        return _images.get(image_id)


# ---------------------------------------------------------------------------
# Settings bridge
# ---------------------------------------------------------------------------

def heightmap_settings_to_dict(s: HeightmapSettings) -> Dict[str, Any]:
    """Convert an API HeightmapSettings model to the dict form service.render() expects."""
    return s.model_dump()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def do_render(
    image_id: str,
    settings: HeightmapSettings,
    profile_name: Optional[str] = None,
    mask_id: Optional[str] = None,
) -> RenderResponse:
    image = get_upload(image_id)
    if image is None:
        raise KeyError(f"Unknown image_id: {image_id!r}")

    # Wizard-supplied subject mask — reuse it instead of re-running
    # inference from settings.subject_mask_backend.
    subject_alpha_override: Optional[np.ndarray] = None
    if mask_id:
        subject_alpha_override = blob_store.load_heightmap(mask_id)
        if subject_alpha_override is None:
            raise KeyError(f"Unknown mask_id: {mask_id!r}")

    profile_data: Dict[str, Any] = {}
    if profile_name:
        profile_data = _load_profile_payload(profile_name)

    merged = merge_profile_settings(profile_data, heightmap_settings_to_dict(settings))
    svc = get_service()

    result: PreviewResult = svc.render(
        image, merged, subject_alpha_override=subject_alpha_override
    )

    heightmap_id = blob_store.store_heightmap(result.heightmap)
    # Store the plain greyscale depth map (not the diagnostic composite panel)
    # so the before/after slider shows just the depth map on the right.
    _gray = Image.fromarray(
        np.round(np.clip(result.heightmap, 0.0, 1.0) * 255).astype(np.uint8), mode="L"
    ).convert("RGB")
    preview_id = blob_store.store_image(_gray, mode="8bit")

    conditioned_id: Optional[str] = None
    if result.conditioned is not None:
        conditioned_id = blob_store.store_image(result.conditioned, mode="8bit")

    render_mask_id: Optional[str] = None
    if result.subject_alpha is not None:
        # subject_alpha is float32 [0,1] (H, W). Persist as a 16-bit PNG so
        # the client can fetch /blob/{id} and use it like any other mask.
        arr16 = (np.clip(result.subject_alpha, 0.0, 1.0) * 65535).astype(np.uint16)
        mask_img = Image.fromarray(arr16, mode="I;16")
        mbuf = io.BytesIO()
        mask_img.save(mbuf, format="PNG")
        render_mask_id = blob_store.store_bytes(mbuf.getvalue(), "image/png")

    # Pre-overlay heightmap — the layered .lbrn2 export engraves this on
    # zone:device so decorative patterns (own layers) aren't burned twice.
    clean_heightmap_id: Optional[str] = None
    if getattr(result, "heightmap_clean", None) is not None:
        clean_heightmap_id = blob_store.store_heightmap(result.heightmap_clean)

    return RenderResponse(
        heightmap_id=heightmap_id,
        preview_id=preview_id,
        elapsed_s=result.elapsed_s,
        image_hash=result.image_hash,
        conditioned_id=conditioned_id,
        render_mask_id=render_mask_id,
        warnings=list(result.warnings),
        clean_heightmap_id=clean_heightmap_id,
    )


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def _gaussian_blur_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return alpha
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(alpha.astype(np.float32), sigma=radius / 3.0)
    except ImportError:
        return alpha


def _get_masker(backend: str) -> Any:
    """Return a cached masker for ``backend``, loading it on first call."""
    with _masker_lock:
        if backend not in _masker_cache:
            masker, _ = load_masker(backend, device="cpu")
            _masker_cache[backend] = masker
        return _masker_cache[backend]


def do_mask(
    image_id: str,
    backend: str,
    edge_softness: float,
) -> MaskResponse:
    image = get_upload(image_id)
    if image is None:
        raise KeyError(f"Unknown image_id: {image_id!r}")

    masker = _get_masker(backend)
    raw = masker.infer(image)
    alpha = raw.alpha if hasattr(raw, "alpha") else np.asarray(raw, dtype=np.float32)
    radius = int(edge_softness * max(image.size) * 0.05)
    alpha = _gaussian_blur_alpha(alpha, radius)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # Store as 16-bit single-channel PNG
    arr16 = (alpha * 65535).astype(np.uint16)
    import PIL.Image as PILImage
    mask_img = PILImage.fromarray(arr16, mode="I;16")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_id = blob_store.store_bytes(buf.getvalue(), "image/png")

    coverage_pct = float(alpha.mean()) * 100.0
    return MaskResponse(mask_id=mask_id, coverage_pct=coverage_pct)


def do_click_mask(
    image_id: str,
    mask_id: Optional[str],
    x: int,
    y: int,
    label: str,
    clicker_key: str,
    tolerance: float,
    max_fraction: float,
) -> MaskResponse:
    image = get_upload(image_id)
    if image is None:
        raise KeyError(f"Unknown image_id: {image_id!r}")

    # Defensive: only use known clicker keys
    effective_key = clicker_key if get_clicker(clicker_key) is not None else DEFAULT_CLICKER_KEY

    if effective_key == DEFAULT_CLICKER_KEY:
        clicker = _FloodFillClicker(tolerance=tolerance, max_fraction=max_fraction)
    else:
        clicker, _ = load_clicker(effective_key, device="cpu")

    if not (0 <= x < image.width and 0 <= y < image.height):
        raise ValueError(
            f"Click coordinate ({x}, {y}) is outside image bounds "
            f"{image.width}x{image.height}"
        )

    int_label = POINT_LABEL_POSITIVE if label == "positive" else POINT_LABEL_NEGATIVE
    grown = clicker.infer(image, [(x, y)], [int_label]).astype(np.float32)
    if grown.ndim != 2:
        raise ValueError(f"Click masker must return a 2-D mask; got shape {grown.shape}")
    h, w = grown.shape

    base = np.zeros((h, w), dtype=np.float32)
    if mask_id:
        existing = blob_store.load_heightmap(mask_id)  # float32 [0,1]
        if existing is not None and existing.shape == (h, w):
            base = existing

    if int_label == POINT_LABEL_POSITIVE:
        merged = np.maximum(base, grown)
    else:
        merged = np.where(grown > 0, 0.0, base).astype(np.float32)

    arr16 = (np.clip(merged, 0, 1) * 65535).astype(np.uint16)
    import PIL.Image as PILImage
    mask_img = PILImage.fromarray(arr16, mode="I;16")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    new_mask_id = blob_store.store_bytes(buf.getvalue(), "image/png")
    return MaskResponse(mask_id=new_mask_id, coverage_pct=float(merged.mean()) * 100.0)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def do_export_png(heightmap_id: str, bit_depth: int = 16) -> bytes:
    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")
    if bit_depth == 16:
        arr = (np.clip(hm, 0, 1) * 65535).astype(np.uint16)
        img = Image.fromarray(arr, mode="I;16")
    else:
        arr = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_subject_mask_to_heightmap(
    hm: np.ndarray,
    mask_arr: np.ndarray,
) -> np.ndarray:
    """Bake a subject mask into a heightmap before per-pass PNG generation.

    mask_arr: float32 (H, W), 1.0 = subject (engrave), 0.0 = background (skip).
    hm:       float32 (H, W), LightBurn polarity: 0.0 = deepest cut, 1.0 = surface (no engraving).

    Uses the same blending formula as the per-pass layer computation so
    background pixels (mask≈0) are raised to 1.0 (no engraving) with a
    smooth gradient at soft edges. Does NOT binarize — rembg/BiRefNet
    soft alpha is preserved directly.
    """
    return np.clip(1.0 - (1.0 - hm) * mask_arr, 0.0, 1.0).astype(np.float32)


def list_profile_names() -> list[str]:
    return list_profiles()


def get_profile_data(name: str) -> Dict[str, Any]:
    return load_profile(name)


# ---------------------------------------------------------------------------
# LBRN2 export
# ---------------------------------------------------------------------------

def do_export_lbrn2(
    plan_id: str,
    heightmap_id: str,
    profile_name: Optional[str] = None,
    subject_mask_id: Optional[str] = None,
    shape_override: Optional[str] = None,
    clean_heightmap_id: Optional[str] = None,
) -> bytes:
    """Serialise a stored PassPlan into a zip bundle ready for LightBurn.

    The zip contains ``project.lbrn2`` plus one ``pass_NN_<kind>.png`` per
    enabled pass. The .lbrn2 references the PNGs by relative filename, so
    when the user unzips the bundle to any directory and opens the
    project, every bitmap layer loads correctly.

    When ``subject_mask_id`` is supplied and the profile is not a coin
    (``shape: circle``), the mask is **baked** into the depth heightmap
    before per-pass PNGs are computed. Background pixels (mask≈0) are
    raised to 1.0 (LightBurn white = no engraving) using the mask's soft
    alpha directly, without binarizing. LightBurn cannot clip via a raster
    PNG — only vector shapes (Ellipse MaskID) can do that. The standalone
    ``subject_mask.png`` ships separately in the bundle zip for LightBurn's
    Trace Image workflow if the user wants an editable vector boundary.
    Coin profiles (``shape: circle``) use a Tool-layer Ellipse MaskID for
    geometric clipping and skip the pixel-baking step.

    Returns the raw zip bytes; the route layer sets the proper content-
    disposition + media-type.
    """
    import io
    import tempfile
    import xml.etree.ElementTree as ET
    import zipfile
    from xml.dom import minidom

    from mopa.lbrn_writer import build_lbrn_tree, make_thumbnail_b64

    plan = get_plan(plan_id)
    if plan is None:
        raise KeyError(f"Unknown plan_id: {plan_id!r}")

    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")

    # Pre-overlay heightmap for the device pass. When the render baked
    # decorative patterns into `hm`, engraving that on zone:device would
    # burn the patterns twice (they get their own layers). Silently absent
    # for old sessions — then `hm` is used as-is.
    hm_clean = blob_store.load_heightmap(clean_heightmap_id) if clean_heightmap_id else None
    if hm_clean is not None and hm_clean.shape != hm.shape:
        hm_clean = None  # resolution mismatch — don't mix scales
    hm_device = hm_clean if hm_clean is not None else hm

    # Resolve the on-bed size. Profile fields ``print_width_mm`` and
    # ``print_height_mm`` define a BOUNDING BOX — the exporter scales
    # the heightmap to fit while preserving its native aspect ratio so
    # a portrait sculptok output never gets stretched into a square
    # plaque profile. The embedded PNG bytes are always full sculptok
    # resolution; only the LightBurn W/H attributes change.
    #
    # Fallback when the profile doesn't supply a box: rasterise at
    # 254 DPI (10 px/mm) so the user gets a sensible mm size without
    # any per-export configuration.
    px_h, px_w = hm.shape
    aspect = float(px_w) / float(px_h) if px_h > 0 else 1.0

    profile_payload: Dict[str, Any] = {}
    if profile_name:
        profile_payload = _load_profile_payload(profile_name)

    starting = profile_payload.get("lightburn_starting_point") or {}
    image_pass_count = int(starting.get("passes", 256))

    box_w = profile_payload.get("print_width_mm")
    box_h = profile_payload.get("print_height_mm")
    box_w = float(box_w) if isinstance(box_w, (int, float)) else None
    box_h = float(box_h) if isinstance(box_h, (int, float)) else None

    if box_w is not None and box_h is not None:
        # Both axes constrained → "contain" fit. Whichever axis is the
        # tighter constraint sets the scale; the other shrinks below
        # the box on its short axis. This is the same fit-to-area
        # logic CSS uses for ``object-fit: contain``.
        scale = min(box_w / float(px_w), box_h / float(px_h))
        print_w_mm = float(px_w) * scale
        print_h_mm = float(px_h) * scale
    elif box_w is not None:
        # Width-only constraint → height follows aspect.
        print_w_mm = box_w
        print_h_mm = box_w / aspect
    elif box_h is not None:
        # Height-only constraint → width follows aspect.
        print_h_mm = box_h
        print_w_mm = box_h * aspect
    else:
        # 254 DPI = 25.4 mm/inch ÷ 254 = 0.1 mm per pixel.
        DEFAULT_DPI = 254.0
        mm_per_px = 25.4 / DEFAULT_DPI
        print_w_mm = float(px_w) * mm_per_px
        print_h_mm = float(px_h) * mm_per_px

    # Centre of the LightBurn work area (175 × 175 mm bed → 87.5 mm each axis).
    # Images and Ellipse shapes are placed at this point so they open centred
    # on the bed without the user having to move them manually.
    _BED_CENTER_MM = 87.5

    # Pad the per-pass PNGs to a square so that any mask shape (circle,
    # rectangle, custom) in LightBurn always has content to clip against.
    # The extra area is filled with 1.0 (white = no engraving) so the laser
    # never fires in the padding region. Physical size of the padded square
    # = the larger of the two contain-fit dimensions.
    sq_mm = max(print_w_mm, print_h_mm)
    sq_px = max(px_w, px_h)

    # Detect coin profile: explicit shape=circle declaration (or shape_override)
    # triggers Tool-layer Ellipse clipping instead of pixel-level mask baking.
    effective_shape = shape_override or profile_payload.get("shape") or "rectangle"
    add_coin_circle = (
        effective_shape == "circle"
        and box_w is not None and box_h is not None
    )

    # Detect zone plan: any zone pass kind present → zone compositing path.
    _ZONE_KINDS = frozenset({PASS_KIND_FIELD, PASS_KIND_BORDER, PASS_KIND_RIM, PASS_KIND_DEVICE, PASS_KIND_EXERGUE})
    is_zone_plan = any(ep.kind in _ZONE_KINDS for ep in plan.passes)

    # Bake subject mask into the heightmap array before any per-pass PNG is
    # computed. This is the only guarantee that background pixels don't fire
    # on the laser — a non-engraving .lbrn2 layer (the old M-layer) provided
    # no such guarantee because it didn't touch the depth data.
    # Coin profiles skip baking: the Ellipse MaskID handles geometric clipping.
    # Zone plans skip global baking too — each zone pass composites independently
    # via zone_hm_for_pass(), and the device zone receives the mask directly.
    if subject_mask_id is not None and not add_coin_circle and not is_zone_plan:
        _mask_arr = blob_store.load_heightmap(subject_mask_id)
        if _mask_arr is not None:
            if _mask_arr.shape != (px_h, px_w):
                _mask_pil = Image.fromarray(
                    (np.clip(_mask_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
                    mode="L",
                ).resize((px_w, px_h), Image.LANCZOS)
                _mask_arr = np.asarray(_mask_pil, dtype=np.float32) / 255.0
            hm = _apply_subject_mask_to_heightmap(hm, _mask_arr)

    # For zone plans, resolve the device zone mask. The stored ep.mask for
    # zone:device is an all-ones placeholder; we swap it here for the actual
    # subject silhouette so only the subject ablates. For zone:field, the
    # device mask is subtracted so the field pass never fires over the
    # subject area (would double-ablate at field parameters).
    #
    # Preferred source: the HEIGHTMAP itself. Sculptok reliefs sit on a flat
    # background, so thresholding it yields a pixel-accurate silhouette —
    # photo-inference masks (rembg/birefnet) segment drawings and line art
    # poorly. The photo mask is only the fallback when the heightmap's
    # background isn't flat enough to threshold.
    device_mask: Optional[np.ndarray] = None
    if is_zone_plan:
        device_mask = mask_from_heightmap(hm_device)
        if device_mask is None and subject_mask_id is not None:
            _dev_raw = blob_store.load_heightmap(subject_mask_id)
            if _dev_raw is not None:
                if _dev_raw.shape != (px_h, px_w):
                    _dev_pil = Image.fromarray(
                        (np.clip(_dev_raw, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
                        mode="L",
                    ).resize((px_w, px_h), Image.LANCZOS)
                    device_mask = np.asarray(_dev_pil, dtype=np.float32) / 255.0
                else:
                    device_mask = _dev_raw
        if device_mask is not None:
            # Harden the split. Soft photo-inference masks (rembg) carry
            # broad 0.3–0.7 regions; splitting zones with those makes BOTH
            # the device and the surrounding zone partially engrave the
            # same pixels (double partial burns — visible smears). Binarise
            # at 0.5, then re-feather a couple of px so the seam doesn't
            # alias.
            device_mask = (device_mask >= 0.5).astype(np.float32)
            try:
                import cv2
                device_mask = cv2.GaussianBlur(device_mask, (5, 5), 0)
            except ImportError:
                pass

    # Settings snapshot from plan time — drives export-time pattern layers.
    plan_settings = get_plan_settings(plan_id)

    # Zone precedence (classic coin composition):
    #   * rings (border/rim) FRAME the design — the device is clipped to the
    #     field region so a subject spilling over the rings is cropped by
    #     them, not the other way round;
    #   * the exergue strip is a reserved text panel — the device is clipped
    #     out of it too (the design "sits on the exergue line");
    #   * the field yields to the device (subject floats on the background).
    # field_geo already excludes rings + exergue, so clipping the device to
    # it implements all three rules at once.
    field_geo: Optional[np.ndarray] = None
    for _ep in plan.passes:
        if _ep.kind == PASS_KIND_FIELD:
            field_geo = _ep.mask.astype(np.float32, copy=False)
            if field_geo.shape != (px_h, px_w):
                _fg_pil = Image.fromarray(
                    (np.clip(field_geo, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L",
                ).resize((px_w, px_h), Image.LANCZOS)
                field_geo = np.asarray(_fg_pil, dtype=np.float32) / 255.0
            break
    if device_mask is not None and field_geo is not None:
        device_mask = np.clip(device_mask * field_geo, 0.0, 1.0)

    # Materialise per-pass PNGs into a scratch dir, then zip them up with
    # the project. Scratch dir is cleaned up before this function returns.
    bundle_dir = Path(tempfile.mkdtemp(prefix="mopa_lbrn2_"))
    try:
        depth_shapes = []
        png_paths: list[tuple[str, Path]] = []
        _ZONE_SHORT = {
            PASS_KIND_FIELD: "field",
            PASS_KIND_BORDER: "border",
            PASS_KIND_RIM: "rim",
        }
        for idx, ep in enumerate(plan.passes):
            png_name = f"pass_{idx:02d}_{ep.kind.replace(':', '_')}.png"
            png_path = bundle_dir / png_name
            if ep.kind in _ZONE_KINDS:
                # Zone compositing: raise out-of-zone pixels to 1.0 (no engraving)
                # then apply bilateral filter to suppress seam halos.
                if ep.kind == PASS_KIND_DEVICE:
                    # Subject mask overrides the all-ones placeholder stored in ep.mask.
                    eff_mask = device_mask if device_mask is not None else np.ones((px_h, px_w), dtype=np.float32)
                elif ep.kind == PASS_KIND_FIELD and device_mask is not None:
                    # The field yields to the device: never re-engrave the
                    # subject area at field parameters.
                    eff_mask = np.clip(ep.mask.astype(np.float32, copy=False) * (1.0 - device_mask), 0.0, 1.0)
                else:
                    eff_mask = ep.mask.astype(np.float32, copy=False)
                # Guard: zone mask was computed at plan time and may differ in
                # resolution from the heightmap loaded at export time.
                if eff_mask.shape != (px_h, px_w):
                    _em_pil = Image.fromarray(
                        (np.clip(eff_mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
                        mode="L",
                    ).resize((px_w, px_h), Image.LANCZOS)
                    eff_mask = np.asarray(_em_pil, dtype=np.float32) / 255.0
                # Decorative zones carry their OWN pattern relief, generated
                # here from the plan-time settings — the layer works even if
                # the stored heightmap was rendered before patterns were
                # chosen. Falls back to masking the sculpt heightmap when no
                # pattern is configured for the zone.
                pattern = None
                if ep.kind in _ZONE_SHORT and plan_settings:
                    pattern = zone_pattern_layer(
                        _ZONE_SHORT[ep.kind], plan_settings, px_h, px_w,
                        print_w_mm=print_w_mm, print_h_mm=print_h_mm,
                    )
                if ep.kind == PASS_KIND_EXERGUE:
                    # Reserved text panel: a flat recess at exergue params —
                    # never leftover sculpt content. Text relief lands here
                    # once exergue text is supported.
                    layer = np.clip(1.0 - eff_mask, 0.0, 1.0).astype(np.float32)
                elif pattern is not None:
                    # Composite the pattern relief against white through the
                    # zone mask: outside the zone nothing engraves.
                    layer = np.clip(
                        1.0 - (1.0 - pattern) * eff_mask, 0.0, 1.0
                    ).astype(np.float32)
                else:
                    # Zone passes engrave the CLEAN sculpt — decorative
                    # patterns baked into `hm` live in their own layers.
                    layer = zone_hm_for_pass(hm_device, eff_mask)
            elif ep.kind == "pre_clean":
                # Pre-clean is a flat low-power surface sweep — a solid fill,
                # not the sculpt depths.
                layer = np.zeros((px_h, px_w), dtype=np.float32)
            else:
                mask = ep.mask.astype(np.float32, copy=False)
                layer = np.clip(1.0 - (1.0 - hm) * mask, 0.0, 1.0).astype(np.float32)
            # Flip vertically: LightBurn's Y axis is downward (top = y=0);
            # the heightmap is stored top-row-first which renders upside-down
            # on the bed. flipud corrects the engraving to match the original photo.
            layer = np.flipud(layer)
            # Pad non-square layers to a square with white (1.0 = no engraving)
            # so any LightBurn mask shape (circle, rectangle) has full coverage.
            if layer.shape[0] != layer.shape[1]:
                pad_h, pad_w = layer.shape
                padded = np.ones((sq_px, sq_px), dtype=np.float32)
                y0 = (sq_px - pad_h) // 2
                x0 = (sq_px - pad_w) // 2
                padded[y0:y0 + pad_h, x0:x0 + pad_w] = layer
                layer = padded
            save_lightburn_png(layer, png_path)
            png_paths.append((png_name, png_path))
            depth_shapes.append(ShapeRef(
                cut_index=ep.cut_setting.index,
                shape_type="Bitmap",
                source_file=png_name,
                source_path=png_path,
                embed_data=True,
                physical_width_mm=sq_mm,
                physical_height_mm=sq_mm,
                center_x_mm=_BED_CENTER_MM,
                center_y_mm=_BED_CENTER_MM,
            ))

        profile = plan.profile
        used = {ep.cut_setting.index: ep.cut_setting for ep in plan.passes}

        # Human-readable layer names — the LightBurn card's generic C00/C07
        # labels tell the user nothing in the Cuts/Layers panel. Custom card
        # names (anything not matching the generic CNN pattern) are kept.
        import re as _re
        _KIND_LAYER_NAMES = {
            "pre_clean": "PreClean",
            "photo_tonal": "PhotoTonal",
            "signature": "Signature",
            "form": "Sculpt3D",
        }
        # Single-sweep passes must not inherit the global 3D-slice pass
        # count (256/512) — one pass unless the card says otherwise.
        _SINGLE_SWEEP_KINDS = {"pre_clean", "photo_tonal", "signature"}
        for ep in plan.passes:
            fe = used.get(ep.cut_setting.index)
            if fe is None:
                continue
            nice = _KIND_LAYER_NAMES.get(ep.kind)
            rename = nice is not None and _re.fullmatch(r"C\d{1,2}", fe.name or "")
            single = ep.kind in _SINGLE_SWEEP_KINDS and "numPasses" not in fe.raw
            if not rename and not single:
                continue
            named_raw = dict(fe.raw)
            if rename:
                named_raw["name"] = nice
            if single:
                named_raw["numPasses"] = "1"
            used[fe.index] = ColorEntry(
                index=fe.index, name=nice if rename else fe.name,
                max_power=fe.max_power, speed=fe.speed, frequency=fe.frequency,
                q_pulse_width=fe.q_pulse_width, interval=fe.interval,
                raw=named_raw,
            )

        # Apply lightburn_starting_point overrides to the form pass so the
        # exported .lbrn2 uses the profile's actual laser parameters rather
        # than the card's generic C01 defaults.
        if starting:
            for ep in plan.passes:
                if ep.kind != "form":
                    continue
                fe = ep.cut_setting
                sp_speed    = starting.get("speed_mm_s")
                sp_power    = starting.get("power_percent")
                sp_freq_khz = starting.get("frequency_khz")
                sp_pulse_ns = starting.get("pulse_width_ns")
                sp_interval = starting.get("line_interval_mm")
                if not any(v is not None for v in [sp_speed, sp_power, sp_freq_khz, sp_pulse_ns, sp_interval]):
                    break
                override_raw = dict(fe.raw)
                if sp_speed is not None:
                    override_raw["speed"] = str(int(float(sp_speed)))
                if sp_power is not None:
                    override_raw["maxPower"] = str(int(float(sp_power)))
                if sp_freq_khz is not None:
                    override_raw["frequency"] = str(int(float(sp_freq_khz) * 1000))
                if sp_pulse_ns is not None:
                    override_raw["QPulseWidth"] = str(int(sp_pulse_ns))
                if sp_interval is not None:
                    override_raw["interval"] = str(float(sp_interval))
                # Don't embed numPasses here — image_pass_count controls it below
                override_raw.pop("numPasses", None)
                # Enable cleanup SubLayer when profile requests it
                if starting.get("cleanup_every_passes"):
                    override_raw["cleanupPass"] = "1"
                used[fe.index] = ColorEntry(
                    index=fe.index,
                    name=fe.name,
                    max_power=float(sp_power) if sp_power is not None else fe.max_power,
                    speed=float(sp_speed) if sp_speed is not None else fe.speed,
                    frequency=int(float(sp_freq_khz) * 1000) if sp_freq_khz is not None else fe.frequency,
                    q_pulse_width=int(sp_pulse_ns) if sp_pulse_ns is not None else fe.q_pulse_width,
                    interval=float(sp_interval) if sp_interval is not None else fe.interval,
                    raw=override_raw,
                )
                break

        if add_coin_circle:
            # Ellipse will be inserted at this position in the final list.
            ellipse_shape_id = len(depth_shapes)
            # Rebuild depth shapes with MaskID pointing at the Ellipse.
            depth_shapes = [
                ShapeRef(
                    cut_index=s.cut_index,
                    shape_type=s.shape_type,
                    source_file=s.source_file,
                    source_path=s.source_path,
                    embed_data=s.embed_data,
                    physical_width_mm=s.physical_width_mm,
                    physical_height_mm=s.physical_height_mm,
                    center_x_mm=s.center_x_mm,
                    center_y_mm=s.center_y_mm,
                    mask_id=ellipse_shape_id,
                )
                for s in depth_shapes
            ]

        shapes = list(depth_shapes)

        if add_coin_circle:
            coin_radius = min(print_w_mm, print_h_mm) / 2.0
            shapes.append(ShapeRef(
                cut_index=30,
                shape_type="Ellipse",
                rx=coin_radius,
                ry=coin_radius,
                used_by_id=0,  # first depth Bitmap's ShapeID
                xform=(1.0, 0.0, 0.0, 1.0, _BED_CENTER_MM, _BED_CENTER_MM),
            ))
            # Tool layer entry — LightBurn's fixed slot 30.
            used[30] = ColorEntry(
                index=30, name="T1",
                max_power=0.0, speed=0.0, frequency=0, q_pulse_width=0, interval=0.0,
                raw={"index": "30", "name": "T1", "priority": "1"},
            )

        import datetime as _dt
        _profile_label = profile_name or "default"
        _ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # --- Breakthrough pass file (optional) ----------------------------
        # When the profile defines a `breakthrough_pass` section, generate a
        # standalone breakthrough.lbrn2 with numPasses=1 at the breakthrough
        # settings. The user runs this file first (with spray applied) to
        # micro-roughen the mirror surface before the main 3D sculpt begins.
        bt_payload = profile_payload.get("breakthrough_pass")
        bt_extra_files: list[tuple[str, bytes]] = []
        if bt_payload:
            bt_speed = float(bt_payload.get("speed_mm_s", 400))
            bt_power = float(bt_payload.get("power_percent", 45))
            bt_freq_khz = float(bt_payload.get("frequency_khz", 45))
            bt_pulse_ns = int(bt_payload.get("pulse_width_ns", 200))
            bt_interval = float(bt_payload.get("line_interval_mm", 0.015))

            bt_png_path = bundle_dir / "breakthrough.png"
            # Solid black = full-area surface scan at breakthrough settings
            save_lightburn_png(np.zeros((sq_px, sq_px), dtype=np.float32), bt_png_path)

            # Use raw dict so numPasses=1 overrides the global image_pass_count
            bt_entry = ColorEntry(
                index=0, name="Breakthrough",
                max_power=bt_power, speed=bt_speed,
                frequency=int(bt_freq_khz * 1000),
                q_pulse_width=bt_pulse_ns, interval=bt_interval,
                raw={
                    "index": "0", "name": "Breakthrough",
                    "maxPower": str(int(bt_power)),
                    "speed": str(int(bt_speed)),
                    "frequency": str(int(bt_freq_khz * 1000)),
                    "QPulseWidth": str(bt_pulse_ns),
                    "interval": str(bt_interval),
                    "numPasses": "1",
                    "ditherMode": "3dslice",
                    "negative": "0",
                    "subname": "Breakthrough",
                    "dpi": "1270",
                },
            )
            bt_shape = ShapeRef(
                cut_index=0, shape_type="Bitmap",
                source_file="breakthrough.png", source_path=bt_png_path,
                embed_data=True,
                physical_width_mm=sq_mm, physical_height_mm=sq_mm,
                center_x_mm=_BED_CENTER_MM, center_y_mm=_BED_CENTER_MM,
            )
            bt_tree = build_lbrn_tree(
                entries=[bt_entry], shapes=[bt_shape],
                notes=(
                    "STEP 1 — Run this file first with spray applied. "
                    "Apply anti-reflection spray evenly, let it dry 30 s, then fire. "
                    "This single pass bonds the spray and micro-roughens the surface "
                    "to break the mirror finish before deep 3D sculpting. "
                    "After completion open project.lbrn2 for the full depth engraving."
                ),
            )
            bt_xml = ET.tostring(bt_tree.getroot(), encoding="utf-8")
            bt_pretty = minidom.parseString(bt_xml).toprettyxml(indent="  ", encoding="utf-8")
            bt_extra_files.append(("breakthrough.lbrn2", bt_pretty))
            bt_extra_files.append(("breakthrough.png", bt_png_path.read_bytes()))

        _bt_prefix = "STEP 2 — Run after breakthrough.lbrn2 | " if bt_payload else ""
        _notes = (
            f"{_bt_prefix}Generated by MOPA Heightmap Studio | "
            f"Profile: {_profile_label} | "
            f"Size: {print_w_mm:.1f}x{print_h_mm:.1f} mm | "
            f"Exported: {_ts}"
        )
        tree = build_lbrn_tree(
            entries=list(used.values()),
            shapes=shapes,
            app_version=profile.app_version or "1.7.00",
            thumbnail_b64=make_thumbnail_b64(hm),
            image_negative=bool(profile_payload.get("polarity_invert", False)),
            image_pass_count=image_pass_count,
            notes=_notes,
        )
        xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.lbrn2", pretty)
            for name, path in png_paths:
                zf.write(path, arcname=name)
            for arcname, data in bt_extra_files:
                zf.writestr(arcname, data)
        return zbuf.getvalue()
    finally:
        # Clean up every file we wrote to the scratch dir, not just the
        # ones that ended up in png_paths (the mask PNG is on disk but not
        # in png_paths because it's embedded into the .lbrn2 instead).
        try:
            for child in bundle_dir.iterdir():
                try:
                    child.unlink()
                except OSError:
                    pass
        except OSError:
            pass
        try:
            bundle_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Pass planner
# ---------------------------------------------------------------------------

# Simple deterministic colour palette for pass entries.
_PASS_PALETTE: List[str] = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c",
    "#3498db", "#9b59b6", "#34495e", "#e91e63", "#607d8b",
]

_plan_lock = threading.Lock()
_plans: Dict[str, Any] = {}            # plan_id -> PassPlan (opaque; used by export)
_plan_settings: Dict[str, Dict[str, Any]] = {}  # plan_id -> settings dict at plan time


def store_plan(plan: Any, settings: Optional[Dict[str, Any]] = None) -> str:
    plan_id = str(uuid.uuid4())
    with _plan_lock:
        _plans[plan_id] = plan
        if settings is not None:
            _plan_settings[plan_id] = dict(settings)
    return plan_id


def get_plan(plan_id: str) -> Optional[Any]:
    with _plan_lock:
        return _plans.get(plan_id)


def get_plan_settings(plan_id: str) -> Dict[str, Any]:
    """Settings snapshot taken when the plan was computed — the exporter
    uses these to generate zone pattern layers without a re-render."""
    with _plan_lock:
        return dict(_plan_settings.get(plan_id, {}))


def _build_zone_passes(
    hm: np.ndarray,
    profile_payload: Dict[str, Any],
    material_profile: Any,
    shape_override: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Optional[List[EngravingPass]]:
    """Build zone EngravingPass objects when the profile has zone_params.

    Returns None when zones are not configured or geometry is missing.
    The device zone gets an all-ones mask (placeholder); the real subject
    mask is applied per-pixel at export time.

    Zones with a decorative pattern selected in ``settings`` are switched
    from the profile's anneal parameters to REAL 3D sculpting: the device's
    lightburn_starting_point parameters with the pass count scaled by the
    pattern's depth slider. A guilloche at 2 annealing passes is a tinted
    stencil; at depth×sculpt-passes it's actual relief.
    """
    zone_params = profile_payload.get("zone_params")
    if not zone_params:
        return None

    box_w = profile_payload.get("print_width_mm")
    box_h = profile_payload.get("print_height_mm")
    if not (isinstance(box_w, (int, float)) and isinstance(box_h, (int, float))):
        return None

    px_h, px_w = hm.shape
    box_w, box_h = float(box_w), float(box_h)
    scale = min(box_w / px_w, box_h / px_h)
    px_per_mm = 1.0 / scale

    shape = shape_override or str(profile_payload.get("shape", "rectangle"))
    if shape not in VALID_SHAPES:
        shape = "rectangle"

    zone_geo = profile_payload.get("zone_geometry") or {}
    sigma_scale = float(profile_payload.get("zone_boundary_sigma_scale", 1.0))

    # When the exergue pass is disabled its strip belongs to the FIELD —
    # otherwise the bottom strip would be a dead zone no pass engraves.
    exergue_on = settings is None or bool(settings.get("exergue_enabled", False))
    geo = zone_masks_from_geometry(
        px_h, px_w,
        shape=shape,
        print_w_mm=box_w,
        print_h_mm=box_h,
        px_per_mm=px_per_mm,
        border_width_mm=float(zone_geo.get("border_width_mm", 1.5)),
        rim_width_mm=float(zone_geo.get("rim_width_mm", 0.5)),
        exergue_height_fraction=(
            float(zone_geo.get("exergue_height_fraction", 0.18)) if exergue_on else 0.0
        ),
        hole_radius_mm=zone_geo.get("hole_radius_mm"),
        sigma_scale=sigma_scale,
    )

    starting = profile_payload.get("lightburn_starting_point") or {}
    cleanup = starting.get("cleanup_every_passes")
    zone_entries = entries_from_zone_params(
        zone_params, starting, cleanup_every_passes=cleanup
    )

    # 3D decoration mode: a zone with a pattern selected sculpts real relief
    # at the device's calibrated parameters. The pattern's depth slider sets
    # relief depth as a fraction of the full sculpt depth via the pass count.
    _DECOR_KEYS = {
        PASS_KIND_FIELD:  ("field_pattern",  "field_pattern_depth",  0.70),
        PASS_KIND_BORDER: ("border_pattern", "border_pattern_depth", 0.85),
        PASS_KIND_RIM:    ("rim_pattern",    "rim_pattern_depth",    1.0),
    }
    if settings and starting:
        base_passes = int(starting.get("passes", 256))
        for kind, (pat_key, depth_key, depth_default) in _DECOR_KEYS.items():
            if str(settings.get(pat_key, "none") or "none").lower() == "none":
                continue
            entry = zone_entries.get(kind)
            if entry is None:
                continue
            depth = float(np.clip(settings.get(depth_key, depth_default), 0.05, 1.0))
            n_passes = max(1, int(round(base_passes * depth)))
            sp_speed = float(starting.get("speed_mm_s", entry.speed))
            sp_power = float(starting.get("power_percent", entry.max_power))
            sp_freq_hz = int(float(starting.get("frequency_khz", entry.frequency / 1000.0)) * 1000)
            sp_pulse = int(starting.get("pulse_width_ns", entry.q_pulse_width))
            sp_interval = float(starting.get("line_interval_mm", entry.interval))
            raw = dict(entry.raw)
            raw.update({
                "speed": str(int(sp_speed)),
                "maxPower": str(int(sp_power)),
                "frequency": str(sp_freq_hz),
                "QPulseWidth": str(sp_pulse),
                "interval": str(sp_interval),
                "numPasses": str(n_passes),
            })
            zone_entries[kind] = ColorEntry(
                index=entry.index, name=entry.name,
                max_power=sp_power, speed=sp_speed, frequency=sp_freq_hz,
                q_pulse_width=sp_pulse, interval=sp_interval, raw=raw,
            )

    _ZONE_KIND_ORDER = [
        PASS_KIND_FIELD,
        PASS_KIND_BORDER,
        PASS_KIND_RIM,
        PASS_KIND_DEVICE,
        PASS_KIND_EXERGUE,
    ]
    _ZONE_GEO_KEY = {
        PASS_KIND_FIELD:   "field",
        PASS_KIND_BORDER:  "border",
        PASS_KIND_RIM:     "rim",
        PASS_KIND_DEVICE:  None,     # all-ones; subject mask applied at export
        PASS_KIND_EXERGUE: "exergue",
    }
    _ZONE_NOTES = {
        PASS_KIND_FIELD:   "Annealing sweep — field background.",
        PASS_KIND_BORDER:  "Border band — decorative ring.",
        PASS_KIND_RIM:     "Rim — outermost raised edge.",
        PASS_KIND_DEVICE:  "Device — subject ablation (fires last).",
        PASS_KIND_EXERGUE: "Exergue — bottom text strip.",
    }

    passes: List[EngravingPass] = []
    for kind in _ZONE_KIND_ORDER:
        entry = zone_entries.get(kind)
        if entry is None:
            continue
        # Exergue is opt-in: without text content the pass would deep-engrave
        # whatever background happens to sit in the bottom strip.
        if (
            kind == PASS_KIND_EXERGUE
            and settings is not None
            and not bool(settings.get("exergue_enabled", False))
        ):
            continue
        geo_key = _ZONE_GEO_KEY[kind]
        if geo_key is not None:
            mask = geo[geo_key]
        else:
            mask = np.ones((px_h, px_w), dtype=np.float32)
        passes.append(EngravingPass(
            id=kind,
            kind=kind,
            name=entry.name,
            mask=mask,
            cut_setting=entry,
            enabled=True,
            note=_ZONE_NOTES[kind],
        ))
    return passes or None


def do_plan(
    image_id: str,
    heightmap_id: str,
    profile_name: Optional[str] = None,
    settings: Optional[HeightmapSettings] = None,
    shape_override: Optional[str] = None,
) -> PassPlanResponse:
    """Plan the engraving stack.

    The depth layer (``form``) is always emitted; refinement passes
    (color clusters, photo-tonal, signature, pre-clean) are opt-in. Color
    masks are computed with LAB k-means on the source image when
    ``settings.n_color_passes`` is set (defaults to 0 = monochrome stack).

    When the profile contains ``zone_params``, zone passes replace the
    ``form`` pass: field → border → rim → device → exergue in fire order.
    """
    from mopa.color_quantize import color_masks_for_planner, quantize_to_color_masks

    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")

    profile_payload = _load_profile_payload(profile_name)
    card_path = _resolve_lightburn_card_path(profile_payload)
    material_profile = load_lightburn_card(card_path)

    color_masks: Dict[str, np.ndarray] = {}
    n_color = int(getattr(settings, "n_color_passes", 0) or 0) if settings else 0
    if n_color >= 2:
        image = get_upload(image_id)
        if image is not None:
            clusters = quantize_to_color_masks(image, k=n_color)
            color_masks = color_masks_for_planner(clusters)

    user_toggles: Dict[str, bool] = {}
    if settings:
        user_toggles["pre_clean"]   = bool(getattr(settings, "pre_clean_enabled", False))
        user_toggles["photo_tonal"] = bool(getattr(settings, "photo_tonal_enabled", False))
        user_toggles["signature"]   = bool((getattr(settings, "signature_text", "") or "").strip())

    # Build zone passes when the profile defines zone_params.
    zone_passes = _build_zone_passes(
        hm, profile_payload, material_profile, shape_override=shape_override,
        settings=heightmap_settings_to_dict(settings) if settings else None,
    )
    if zone_passes:
        # Zones replace the form pass — disable it.
        user_toggles[PASS_KIND_FORM] = False

    result = _plan_passes(
        heightmap=hm,
        profile=material_profile,
        user_toggles=user_toggles,
        mask_per_color=color_masks,
        kind_color_overrides=_profile_kind_color_overrides(profile_payload),
    )

    if zone_passes:
        # Inject zone passes in fire order between pre_clean and photo_tonal/signature.
        pre_clean = [p for p in result.passes if p.kind == "pre_clean"]
        refinement = [p for p in result.passes if p.kind not in (
            "pre_clean", PASS_KIND_FORM,
        )]
        merged = tuple(pre_clean + zone_passes + refinement)
        result = PassPlan(profile=result.profile, passes=merged)

    plan_id = store_plan(
        result,
        settings=heightmap_settings_to_dict(settings) if settings else None,
    )
    per_pass_depth = 0.0

    _KIND_LABELS: Dict[str, str] = {
        "pre_clean":    "Pre-clean — oxidation burn-off",
        "form":         "Form — 3D depth carve",
        "photo_tonal":  "Photo-tonal — luminance overlay",
        "signature":    "Signature",
        "zone:field":   "Zone: Field — background annealing",
        "zone:border":  "Zone: Border — decorative band",
        "zone:rim":     "Zone: Rim — outer edge",
        "zone:device":  "Zone: Device — subject ablation",
        "zone:exergue": "Zone: Exergue — bottom text strip",
    }

    def _pass_label(p: EngravingPass) -> str:
        base = _KIND_LABELS.get(p.kind)
        if base:
            return base
        # color:CXX passes
        if p.kind.startswith("color:"):
            return f"Color — {p.cut_setting.name}"
        return p.kind

    entries = [
        PassEntry(
            pass_number=p.cut_setting.index,
            label=_pass_label(p),
            depth_um=per_pass_depth,
            color_hex=_PASS_PALETTE[p.cut_setting.index % len(_PASS_PALETTE)],
        )
        for p in result.passes
    ]
    # Runtime estimate using profile physical size and pass count.
    try:
        _starting = profile_payload.get("lightburn_starting_point", {})
        _pass_count = int(_starting.get("passes", 256))
        _w_mm = float(profile_payload.get("print_width_mm", 30.0))
        _h_mm = float(profile_payload.get("print_height_mm", 30.0))
        _burn = estimate_burn_time(
            result,
            width_mm=_w_mm,
            height_mm=_h_mm,
            pass_count_overrides={p.kind: _pass_count for p in result.passes},
        )
        _runtime_s = _burn.total_seconds
    except Exception:
        _runtime_s = 0.0

    return PassPlanResponse(plan_id=plan_id, passes=entries, estimated_runtime_s=_runtime_s)
