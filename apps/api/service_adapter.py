"""Thin adapter: wraps zoedepth.laser.* to serve the FastAPI routes.

NO business logic lives here — only the glue between HTTP schemas and the
existing Python service objects.  All math stays in zoedepth.laser.*.
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
from zoedepth.laser.service import (
    DEFAULT_SETTINGS,
    HeightmapService,
    PreviewResult,
    merge_profile_settings,
)
from zoedepth.laser.settings import AppSettings, load_settings
from zoedepth.laser.subject_mask import load_masker, list_maskers
from zoedepth.laser.click_mask import (
    DEFAULT_CLICKER_KEY,
    DEFAULT_FLOOD_TOLERANCE,
    DEFAULT_FLOOD_MAX_FRACTION,
    POINT_LABEL_POSITIVE,
    POINT_LABEL_NEGATIVE,
    get_clicker,
    load_clicker,
    _FloodFillClicker,  # type: ignore[attr-defined]
)
from zoedepth.laser.profiles import (
    list_profiles,
    load_profile,
    get_user_profiles_dir,
    validate_profile,
)
from zoedepth.laser.exporter import hash_image, save_lightburn_png, save_master16_png
from zoedepth.laser.heightmap import to_uint16
from zoedepth.laser.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    load_lightburn_card,
)
from zoedepth.laser.lbrn_writer import build_lbrn_tree, ShapeRef
from zoedepth.laser.stages import plan_passes as _plan_passes

from . import blob_store
from .schemas import (
    HeightmapSettings,
    MaskResponse,
    PassEntry,
    PassPlanResponse,
    RenderResponse,
    UploadResponse,
)

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
    """Store an uploaded PIL image; return its id + metadata."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    image_id = sha[:40]
    with _img_lock:
        if image_id not in _images:
            _images[image_id] = image.convert("RGB")
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
        try:
            profile_data = load_profile(profile_name)
        except Exception:
            profile_data = {}

    merged = merge_profile_settings(profile_data, heightmap_settings_to_dict(settings))
    svc = get_service()

    result: PreviewResult = svc.render(image, merged)

    heightmap_id = blob_store.store_heightmap(result.heightmap)
    preview_id = blob_store.store_image(result.preview_image, mode="8bit")

    return RenderResponse(
        heightmap_id=heightmap_id,
        preview_id=preview_id,
        elapsed_s=result.elapsed_s,
        image_hash=result.image_hash,
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

    int_label = POINT_LABEL_POSITIVE if label == "positive" else POINT_LABEL_NEGATIVE
    grown = clicker.infer(image, [(x, y)], [POINT_LABEL_POSITIVE]).astype(np.float32)
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
) -> bytes:
    """Serialise a stored PassPlan into a zip bundle ready for LightBurn.

    The zip contains ``project.lbrn2`` plus one ``pass_NN_<kind>.png`` per
    enabled pass. The .lbrn2 references the PNGs by relative filename, so
    when the user unzips the bundle to any directory and opens the
    project, every bitmap layer loads correctly.

    Returns the raw zip bytes; the route layer sets the proper content-
    disposition + media-type.
    """
    import io
    import tempfile
    import xml.etree.ElementTree as ET
    import zipfile
    from xml.dom import minidom

    from zoedepth.laser.lbrn_writer import build_lbrn_tree
    from zoedepth.laser.exporter import save_master16_png

    plan = get_plan(plan_id)
    if plan is None:
        raise KeyError(f"Unknown plan_id: {plan_id!r}")

    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")

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
            shapes.append(ShapeRef(
                cut_index=ep.cut_setting.index,
                shape_type="Bitmap",
                source_file=png_name,
            ))

        profile = plan.profile
        used = {ep.cut_setting.index: ep.cut_setting for ep in plan.passes}
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
        for _, path in png_paths:
            try:
                path.unlink(missing_ok=True)
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
    """Plan the engraving stack with real per-pass masks.

    The heightmap supplies form / cleanup / detail / shading / polish masks
    via :func:`zoedepth.laser.pass_masks.derive_pass_masks`. Color masks
    are computed with LAB k-means on the source image when ``settings``
    requests them via ``n_color_passes`` (passed through as a custom
    field on a future schema; defaults to 0 = monochrome stack).
    """
    from zoedepth.laser.color_quantize import color_masks_for_planner, quantize_to_color_masks
    from zoedepth.laser.pass_masks import derive_pass_masks
    from zoedepth.laser.stages import PASS_KIND_FORM

    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")

    card_stem = (profile_name or DEFAULT_PROFILE_NAME).removesuffix(".lbrn2")
    card_path = DEFAULT_CARDS_DIR / f"{card_stem}.lbrn2"
    if not card_path.exists():
        card_path = DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2"
    material_profile = load_lightburn_card(card_path)

    kind_masks = derive_pass_masks(hm)
    color_masks: Dict[str, np.ndarray] = {}
    n_color = int(getattr(settings, "n_color_passes", 0) or 0) if settings else 0
    if n_color >= 2:
        image = get_upload(image_id)
        if image is not None:
            clusters = quantize_to_color_masks(
                image, k=n_color, subject_mask=kind_masks.get(PASS_KIND_FORM),
            )
            color_masks = color_masks_for_planner(clusters)

    result = _plan_passes(
        heightmap=hm,
        profile=material_profile,
        masks=kind_masks,
        mask_per_color=color_masks,
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
