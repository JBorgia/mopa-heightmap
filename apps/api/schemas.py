"""Pydantic schemas — mirror InferenceSettings / DEFAULT_SETTINGS field-for-field.

Every field that exists in zoedepth.laser.service.DEFAULT_SETTINGS or
zoedepth.laser.settings.InferenceSettings has a corresponding field here with
the identical default so the API is always in sync.  The CI drift-guard
(``apps/api/export_openapi.py``) generates openapi.json from these models and
a git-diff check fails the build if the generated TypeScript types in
``apps/web/src/app/core/api/generated/api.d.ts`` are stale.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Heightmap pipeline settings (mirrors DEFAULT_SETTINGS in service.py)
# ---------------------------------------------------------------------------

class HeightmapSettings(BaseModel):
    near_percentile: float = Field(5.0, ge=0.0, le=50.0)
    far_percentile: float = Field(95.0, ge=50.0, le=100.0)
    gamma: float = Field(0.72, ge=0.1, le=5.0)
    contrast: float = Field(1.0, ge=0.1, le=5.0)
    midtone_boost: float = Field(0.0, ge=-0.5, le=0.5)
    deep_limit: float = Field(0.04, ge=0.0, le=0.5)
    surface_limit: float = Field(0.96, ge=0.5, le=1.0)
    black_is_deep: bool = True
    flatten_background: bool = False
    background_threshold: float = Field(0.88, ge=0.0, le=1.0)
    background_value: float = Field(1.0, ge=0.0, le=1.0)
    smooth: Literal["none", "off", "bilateral", "gaussian"] = "bilateral"
    smooth_diameter: int = Field(9, ge=1, le=50)
    smooth_strength: float = Field(0.08, ge=0.0, le=1.0)
    sharpen: float = Field(0.2, ge=0.0, le=2.0)
    sharpen_sigma: float = Field(2.0, ge=0.1, le=20.0)
    # Stage A — input conditioning
    input_white_balance: bool = False
    input_clahe: bool = False
    input_clahe_clip: float = Field(2.0, ge=0.5, le=10.0)
    input_clahe_grid: int = Field(8, ge=2, le=32)
    input_denoise: bool = False
    input_denoise_strength: float = Field(5.0, ge=0.1, le=30.0)
    input_remove_specular: bool = False
    input_specular_threshold: int = Field(245, ge=128, le=255)
    input_max_dim: int = Field(0, ge=0, le=8192)
    # Stage C extras
    edge_refine: bool = False
    edge_refine_diameter: int = Field(9, ge=1, le=50)
    edge_refine_sigma_color: float = Field(0.08, ge=0.0, le=1.0)
    edge_refine_sigma_space: float = Field(6.0, ge=0.1, le=30.0)
    dither: bool = False
    dither_levels: int = Field(256, ge=2, le=1024)
    target_depth_um: float = Field(0.0, ge=0.0, le=5000.0)
    posterize_passes: int = Field(0, ge=0, le=4096)
    # Stage B — photo-detail injection
    detail_mode: Literal["off", "luminance", "highpass", "both"] = "off"
    detail_strength: float = Field(0.10, ge=0.0, le=1.0)
    detail_highpass_radius: int = Field(9, ge=1, le=50)
    detail_subject_mask: bool = True
    detail_invert: bool = False
    # Phase 2 — subject isolation (hard background flatten)
    subject_mask_enabled: bool = False
    subject_mask_backend: str = "rembg"
    subject_mask_feather_px: int = Field(3, ge=0, le=64)
    subject_mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    # Phase 3 — bulk-depth + FC-integrated normals composite
    relief_enabled: bool = False
    relief_strength: float = Field(0.3, ge=0.0, le=1.0)
    relief_normals_backend: str = "finite_diff"
    relief_pad_fraction: float = Field(0.25, ge=0.0, lt=1.0)
    # Phase 3b — gradient-domain depth compression (Kerber)
    depth_unsharp_enabled: bool = False
    depth_unsharp_gamma: float = Field(0.7, gt=0.0, le=1.0)
    depth_unsharp_blend: float = Field(0.5, ge=0.0, le=1.0)
    # Phase 4 — face-aware per-region depth weighting
    face_relief_enabled: bool = False
    face_relief_strength: float = Field(1.0, ge=0.0, le=2.0)
    # Auto-orient via face landmarks (rotate so eyes are level).
    auto_orient_face: bool = False
    # Marigold-IID-Appearance delighting (CC-BY-NC-4.0 weights, opt-in).
    delight_enabled: bool = False
    delight_backend: str = "marigold_iid"
    # Photo-guided bilateral cross-filter on the raw depth.
    depth_bilateral_enabled: bool = False
    depth_bilateral_diameter: int = Field(9, ge=1, le=50)
    depth_bilateral_sigma_color: float = Field(0.05, ge=0.0, le=1.0)
    depth_bilateral_sigma_space: float = Field(8.0, ge=0.1, le=50.0)
    # Signature pass content.
    signature_text: str = ""
    signature_corner: Literal["tl", "tr", "bl", "br"] = "br"
    signature_height_fraction: float = Field(0.04, ge=0.005, le=0.5)
    signature_margin_fraction: float = Field(0.03, ge=0.0, le=0.5)
    signature_depth_fraction: float = Field(0.6, ge=0.0, le=1.0)
    # Pre-depth super-resolution.
    pre_upscale_enabled: bool = False
    pre_upscale_resolver: str = "lanczos"
    pre_upscale_target_long_side: int = Field(1024, ge=64, le=8192)
    # External heightmap input (sculptok/meshy bring-your-own-relief mode).
    external_heightmap_path: str = ""
    external_heightmap_polarity: Literal["bright_raised", "dark_raised", "auto"] = "bright_raised"
    external_heightmap_auto_stretch: bool = True
    external_heightmap_use_subject_mask: bool = True
    external_heightmap_resampler: str = "realesrgan-x4plus"
    # Relief stylization (ControlNet-Depth + diffusion + bas-relief prompt).
    relief_stylize_enabled: bool = False
    relief_stylize_backend: str = "sdxl_controlnet_depth"
    relief_stylize_steps: int = Field(25, ge=4, le=200)
    relief_stylize_guidance: float = Field(7.5, ge=0.0, le=20.0)
    relief_stylize_controlnet_strength: float = Field(0.95, ge=0.0, le=2.0)
    relief_stylize_seed: int = Field(0, ge=0, le=2_147_483_647)
    relief_stylize_blend: float = Field(1.0, ge=0.0, le=1.0)


class InferenceConfig(BaseModel):
    model_name: str = "ZoeD_NK"
    device: Optional[str] = None        # None -> resolve from app settings
    pad_input: bool = True
    with_flip_aug: bool = True
    tile_size: int = 0
    tile_overlap: int = 128
    precision: Optional[str] = None     # None -> resolve from app settings
    inference_resolution: int = 0       # 0 = full


# ---------------------------------------------------------------------------
# Upload / blob
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    image_id: str
    w: int
    h: int
    sha256: str


class BlobInfo(BaseModel):
    blob_id: str
    content_type: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    image_id: str
    settings: HeightmapSettings = Field(default_factory=HeightmapSettings)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    profile_name: Optional[str] = None


class RenderResponse(BaseModel):
    heightmap_id: str       # blob id — fetch via GET /blob/{id}
    preview_id: str         # shaded preview blob id
    elapsed_s: float
    image_hash: str


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

class MaskRequest(BaseModel):
    image_id: str
    backend: str = "rembg"
    edge_softness: float = Field(0.0, ge=0.0, le=1.0)


class MaskResponse(BaseModel):
    mask_id: str            # blob id — 16-bit PNG alpha, same dims as upload
    coverage_pct: float


class ClickMaskRequest(BaseModel):
    image_id: str
    mask_id: Optional[str] = None
    x: int
    y: int
    label: Literal["positive", "negative"] = "positive"
    clicker_key: str = "flood-fill"
    tolerance: float = Field(0.08, ge=0.0, le=0.5)
    max_fraction: float = Field(0.6, ge=0.05, le=1.0)


# ---------------------------------------------------------------------------
# Plan / export
# ---------------------------------------------------------------------------

class PassPlanRequest(BaseModel):
    image_id: str
    heightmap_id: str
    profile_name: Optional[str] = None
    settings: HeightmapSettings = Field(default_factory=HeightmapSettings)


class PassEntry(BaseModel):
    pass_number: int
    label: str
    depth_um: float
    color_hex: str


class PassPlanResponse(BaseModel):
    plan_id: str
    passes: List[PassEntry]


class ExportPngRequest(BaseModel):
    heightmap_id: str
    bit_depth: Literal[8, 16] = 16


class ExportLbrn2Request(BaseModel):
    plan_id: str
    heightmap_id: str
    profile_name: Optional[str] = None


class ExportStlRequest(BaseModel):
    heightmap_id: str
    z_scale_mm: float = Field(5.0, ge=0.1, le=100.0)
    base_thickness_mm: float = Field(2.0, ge=0.0, le=50.0)


# ---------------------------------------------------------------------------
# Profile CRUD
# ---------------------------------------------------------------------------

class ProfileSummary(BaseModel):
    name: str
    machine: Optional[str] = None
    lightburn_mode: Optional[str] = None


class ProfileDetail(BaseModel):
    name: str
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Session WebSocket events
# ---------------------------------------------------------------------------

class WsEvent(BaseModel):
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

class ApiError(BaseModel):
    code: str
    message: str
    hint: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ApiError
