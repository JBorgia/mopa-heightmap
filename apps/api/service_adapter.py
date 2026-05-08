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
    load_lightburn_card,
)
from mopa.lbrn_writer import build_lbrn_tree, ShapeRef
from mopa.stages import plan_passes as _plan_passes

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
) -> RenderResponse:
    image = get_upload(image_id)
    if image is None:
        raise KeyError(f"Unknown image_id: {image_id!r}")

    profile_data: Dict[str, Any] = {}
    if profile_name:
        profile_data = _load_profile_payload(profile_name)

    merged = merge_profile_settings(profile_data, heightmap_settings_to_dict(settings))
    svc = get_service()

    result: PreviewResult = svc.render(image, merged)

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

    return RenderResponse(
        heightmap_id=heightmap_id,
        preview_id=preview_id,
        elapsed_s=result.elapsed_s,
        image_hash=result.image_hash,
        conditioned_id=conditioned_id,
        render_mask_id=render_mask_id,
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


def do_mask(
    image_id: str,
    backend: str,
    edge_softness: float,
) -> MaskResponse:
    image = get_upload(image_id)
    if image is None:
        raise KeyError(f"Unknown image_id: {image_id!r}")

    masker, _ = load_masker(backend, device="cpu")
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
) -> bytes:
    """Serialise a stored PassPlan into a zip bundle ready for LightBurn.

    The zip contains ``project.lbrn2`` plus one ``pass_NN_<kind>.png`` per
    enabled pass. The .lbrn2 references the PNGs by relative filename, so
    when the user unzips the bundle to any directory and opens the
    project, every bitmap layer loads correctly.

    When ``subject_mask_id`` is supplied, the mask is added as an
    additional Bitmap shape on a non-engraving layer (Output=0) so the
    user can see + toggle it in LightBurn without it firing by accident.

    Returns the raw zip bytes; the route layer sets the proper content-
    disposition + media-type.
    """
    import io
    import tempfile
    import xml.etree.ElementTree as ET
    import zipfile
    from xml.dom import minidom

    from mopa.lbrn_writer import build_lbrn_tree
    from mopa.exporter import save_master16_png

    plan = get_plan(plan_id)
    if plan is None:
        raise KeyError(f"Unknown plan_id: {plan_id!r}")

    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")

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

    # Materialise per-pass PNGs into a scratch dir, then zip them up with
    # the project. Scratch dir is cleaned up before this function returns.
    bundle_dir = Path(tempfile.mkdtemp(prefix="mopa_lbrn2_"))
    try:
        shapes = []
        png_paths: list[tuple[str, Path]] = []
        for idx, ep in enumerate(plan.passes):
            png_name = f"pass_{idx:02d}_{ep.kind.replace(':', '_')}.png"
            png_path = bundle_dir / png_name
            mask = ep.mask.astype(np.float32, copy=False)
            layer = np.clip(1.0 - (1.0 - hm) * mask, 0.0, 1.0).astype(np.float32)
            save_master16_png(layer, png_path)
            png_paths.append((png_name, png_path))
            # Embed the PNG bytes as base64 inside the .lbrn2 — without
            # this the writer falls through to a stub Shape with no
            # SourceFile and LightBurn can't load the bitmap. The
            # source_file field on ShapeRef is kept for legacy callers
            # but the writer's only working path is embed_data + source_path.
            shapes.append(ShapeRef(
                cut_index=ep.cut_setting.index,
                shape_type="Bitmap",
                source_path=png_path,
                embed_data=True,
                physical_width_mm=print_w_mm,
                physical_height_mm=print_h_mm,
            ))

        profile = plan.profile
        used = {ep.cut_setting.index: ep.cut_setting for ep in plan.passes}

        # Subject mask as a non-engraving layer. Output=0 keeps LightBurn
        # from firing it; numPasses=0 is a belt-and-braces second guard.
        # The user can toggle the layer's visibility/output to use the
        # mask as a guide (manual vector cuts inside the silhouette,
        # secondary anneal pass, etc.) without losing it as a deliverable.
        if subject_mask_id:
            mask_bytes = blob_store.load_bytes(subject_mask_id)
            if mask_bytes is not None:
                # Resize the mask to match the heightmap pixel dimensions
                # before embedding. Subject masks come from /mask in
                # source-photo pixel space (e.g. 539×360), while the
                # heightmap is in sculptok space (e.g. 960×1280) — usually
                # a different aspect ratio. Without resizing, the writer
                # computes a different XForm scale for each shape and the
                # mask renders stretched relative to the depth layer in
                # LightBurn. Aspect-preserving "contain" fit keeps the
                # mask silhouette correctly aligned.
                _orig = Image.open(io.BytesIO(mask_bytes))
                _orig.load()
                if _orig.size != (px_w, px_h):
                    fit_scale = min(px_w / _orig.width, px_h / _orig.height)
                    fit_w = max(1, int(round(_orig.width * fit_scale)))
                    fit_h = max(1, int(round(_orig.height * fit_scale)))
                    _resized = _orig.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
                    # Pad to exact heightmap dims so the XForm math matches
                    # the depth pass — black surrounds the mask silhouette,
                    # which keeps "no engrave outside the subject" semantics.
                    _padded = Image.new(_resized.mode, (px_w, px_h), color=0)
                    paste_x = (px_w - fit_w) // 2
                    paste_y = (px_h - fit_h) // 2
                    _padded.paste(_resized, (paste_x, paste_y))
                    mask_buf = io.BytesIO()
                    _padded.save(mask_buf, format="PNG")
                    mask_bytes = mask_buf.getvalue()
                # Write the (resized) mask PNG to the scratch dir so the
                # writer can read it for embedding. DON'T append it to
                # ``png_paths``: the mask is embedded as base64 inside the
                # .lbrn2 (embed_data=True below), and the bundle endpoint
                # writes a standalone ``subject_mask.png`` from the
                # ORIGINAL (un-resized) blob as a reference artifact, so
                # the user gets the source-resolution mask too.
                mask_path = bundle_dir / "subject_mask.png"
                mask_path.write_bytes(mask_bytes)

                from mopa.lightburn_cards import ColorEntry as _ColorEntry
                # Pick the next free index — above any existing plan-pass
                # index to avoid collisions with depth / photo-tonal /
                # signature layers.
                mask_index = max([99, *(used.keys())]) + 1 if used else 99
                # Clone the depth pass's CutSetting structure verbatim and
                # only flip the fields that need to be different. Building
                # the CutSetting from scratch is fragile — LightBurn 1.7
                # crashes when fields like ``bidir`` / ``priority`` /
                # ``tabCount`` / ``tabCountMax`` are missing, when
                # ``numPasses`` is 0, or when ``subname`` is not one of
                # the known values. Cloning guarantees the XML schema
                # matches LightBurn's expectations.
                depth_pass = plan.passes[0] if plan.passes else None
                depth_raw = (
                    dict(depth_pass.cut_setting.raw)
                    if depth_pass and depth_pass.cut_setting.raw
                    else {}
                )
                # Overrides — index / name move the layer, maxPower=0 and
                # output=0 keep it from firing if the user accidentally
                # enables it, name keeps it identifiable in the LightBurn
                # layer list.
                mask_raw = dict(depth_raw)
                mask_raw.update({
                    "index": str(mask_index),
                    "name": f"M{mask_index:02d}_subject_mask",
                    "maxPower": "0",
                    "maxPower2": "0",
                    "output": "0",
                })
                mask_entry = _ColorEntry(
                    index=mask_index,
                    name=f"M{mask_index:02d}",
                    max_power=0.0,
                    speed=depth_pass.cut_setting.speed if depth_pass else 1000.0,
                    frequency=depth_pass.cut_setting.frequency if depth_pass else 20000,
                    q_pulse_width=depth_pass.cut_setting.q_pulse_width if depth_pass else 100,
                    interval=depth_pass.cut_setting.interval if depth_pass else 0.1,
                    raw=mask_raw,
                )
                used[mask_index] = mask_entry
                shapes.append(ShapeRef(
                    cut_index=mask_index,
                    shape_type="Bitmap",
                    source_path=mask_path,
                    embed_data=True,
                    physical_width_mm=print_w_mm,
                    physical_height_mm=print_h_mm,
                ))

        tree = build_lbrn_tree(
            entries=list(used.values()),
            shapes=shapes,
            app_version=profile.app_version or "1.2.04",
        )
        xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.lbrn2", pretty)
            for name, path in png_paths:
                zf.write(path, arcname=name)
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
_plans: Dict[str, Any] = {}   # plan_id -> PassPlan (opaque; used by export)


def store_plan(plan: Any) -> str:
    plan_id = str(uuid.uuid4())
    with _plan_lock:
        _plans[plan_id] = plan
    return plan_id


def get_plan(plan_id: str) -> Optional[Any]:
    with _plan_lock:
        return _plans.get(plan_id)


def do_plan(
    image_id: str,
    heightmap_id: str,
    profile_name: Optional[str] = None,
    settings: Optional[HeightmapSettings] = None,
) -> PassPlanResponse:
    """Plan the engraving stack.

    The depth layer (``form``) is always emitted; refinement passes
    (color clusters, photo-tonal, signature, pre-clean) are opt-in. Color
    masks are computed with LAB k-means on the source image when
    ``settings.n_color_passes`` is set (defaults to 0 = monochrome stack).
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

    result = _plan_passes(
        heightmap=hm,
        profile=material_profile,
        mask_per_color=color_masks,
        kind_color_overrides=_profile_kind_color_overrides(profile_payload),
    )
    plan_id = store_plan(result)

    n = len(result.passes)
    per_pass_depth = 0.0  # depth_um set by LightBurn cut settings, not service

    entries = [
        PassEntry(
            pass_number=p.cut_setting.index,
            label=f"{p.kind}: {p.cut_setting.name}",
            depth_um=per_pass_depth,
            color_hex=_PASS_PALETTE[p.cut_setting.index % len(_PASS_PALETTE)],
        )
        for p in result.passes
    ]
    return PassPlanResponse(plan_id=plan_id, passes=entries)
