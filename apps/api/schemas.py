"""Pydantic schemas — mirror DEFAULT_SETTINGS in mopa.service.

Every field that exists in mopa.service.DEFAULT_SETTINGS has a
corresponding field here with the identical default so the API stays in
sync. The CI drift-guard (``apps/api/export_openapi.py``) generates
openapi.json from these models and a git-diff check fails the build if
the generated TypeScript types in
``apps/web/src/app/core/api/generated/api.d.ts`` are stale.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Heightmap pipeline settings (mirrors DEFAULT_SETTINGS in service.py)
# ---------------------------------------------------------------------------

class HeightmapSettings(BaseModel):
    # Stage A — pre-sculptok input conditioning
    input_white_balance: bool = False
    input_clahe: bool = False
    input_clahe_clip: float = Field(2.0, ge=0.5, le=10.0)
    input_clahe_grid: int = Field(8, ge=2, le=32)
    input_denoise: bool = False
    input_denoise_strength: float = Field(5.0, ge=0.1, le=30.0)
    input_remove_specular: bool = False
    input_specular_threshold: int = Field(245, ge=128, le=255)
    input_max_dim: int = Field(0, ge=0, le=8192)
    input_auto_orient_face: bool = False
    input_auto_crop: bool = False
    input_auto_crop_aspect: float = Field(0.0, ge=0.0, le=10.0)
    input_auto_crop_prefer_face: bool = True

    # External heightmap source (required at render time).
    external_heightmap_path: str = ""
    external_heightmap_polarity: Literal["bright_raised", "dark_raised", "auto"] = "bright_raised"

    # Polarity invert — flips the saved heightmap so the subject engraves
    # deep instead of the background. Used for signet rings and recessed
    # designs.
    polarity_invert: bool = False

    # Subject mask deliverable (separate artifact, not applied to heightmap).
    subject_mask_enabled: bool = False
    subject_mask_backend: str = "rembg"
    subject_mask_feather_px: int = Field(3, ge=0, le=64)
    subject_mask_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # Procedural background generator — composites a pattern over the
    # photo's background pixels (where the subject mask is 0) BEFORE
    # the photo is sent to sculptok. ``"none"`` disables.
    background_pattern: Literal[
        "none", "guilloche", "stripes", "dots", "halftone", "checkers",
        "solid_black", "solid_white", "solid_grey",
    ] = "none"
    background_scale: float = Field(1.0, ge=0.05, le=20.0)
    background_angle: float = Field(0.0, ge=-180.0, le=180.0)
    background_seed: int = Field(0, ge=0, le=2_147_483_647)
    background_intensity: float = Field(0.6, ge=0.0, le=1.0)

    # LightBurn 3D Sliced polarity convention.
    black_is_deep: bool = True
    background_value: float = Field(1.0, ge=0.0, le=1.0)

    # Heightmap output dither (collapsing 16-bit master to 8-bit cleanly).
    dither: bool = False
    dither_levels: int = Field(256, ge=2, le=1024)

    # Pre-clean pass — defocused full-frame oxide / oil burn-off. Opt-in.
    pre_clean_enabled: bool = False

    # Photo-tonal pass — low-power dithered photographic-luminance overlay.
    photo_tonal_enabled: bool = False
    photo_tonal_invert: bool = False
    photo_tonal_dither: bool = True
    photo_tonal_levels: int = Field(32, ge=2, le=1024)
    photo_tonal_strength: float = Field(0.7, ge=0.0, le=1.0)
    photo_tonal_depth_fraction: float = Field(0.4, ge=0.0, le=1.0)

    # Signature pass content.
    signature_text: str = ""
    signature_corner: Literal["tl", "tr", "bl", "br"] = "br"
    signature_height_fraction: float = Field(0.04, ge=0.005, le=0.5)
    signature_margin_fraction: float = Field(0.03, ge=0.0, le=0.5)
    signature_depth_fraction: float = Field(0.6, ge=0.0, le=1.0)


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
    profile_name: Optional[str] = None


class RenderResponse(BaseModel):
    heightmap_id: str       # blob id — fetch via GET /blob/{id}
    preview_id: str         # shaded preview blob id
    elapsed_s: float
    image_hash: str
    # Photo after pre-sculptok prep (CLAHE/denoise/specular/auto-orient/
    # auto-crop + procedural background composite). Lets the wizard show
    # "this is what sculptok actually saw" without re-running conditioning
    # on the client.
    conditioned_id: Optional[str] = None
    # Subject mask computed during render (only when subject_mask_enabled).
    # Separate from the user-driven /mask flow which has its own mask_id.
    render_mask_id: Optional[str] = None


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


class ExportBundleRequest(BaseModel):
    """Multi-format export → single zip the wizard's "Submit" releases.

    The ``include_*`` flags only gate the heavy / opinionated outputs
    (PNG, .lbrn2, STL). Auxiliary artifacts the server happens to have
    on hand (subject mask, source photo, sculptok input, profile YAML)
    are ALWAYS bundled when supplied — keeping them out would make the
    user re-run the whole wizard to recover a forgotten file.
    """

    heightmap_id: str
    plan_id: Optional[str] = None
    profile_name: Optional[str] = None
    include_png: bool = True
    include_lbrn2: bool = True
    include_stl: bool = True
    # Forwarded to the underlying STL exporter.
    z_scale_mm: float = Field(5.0, ge=0.1, le=100.0)
    base_thickness_mm: float = Field(2.0, ge=0.0, le=50.0)

    # Optional reference artifacts. Each is a blob_id the server resolves
    # via /blob/{id} and writes into the zip under a stable filename.
    # All optional — the bundle endpoint silently skips any that 404.
    image_id: Optional[str] = None              # source photo
    sculptok_input_id: Optional[str] = None     # post-prep photo uploaded to sculptok
    subject_mask_id: Optional[str] = None       # subject mask (deliverable)


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


class ProfileSaveRequest(BaseModel):
    """POST /profiles body — save the current settings as a named profile.

    The server merges any explicitly-supplied fields with sensible
    defaults; in practice the UI just sends ``{name, settings}`` and
    the server fills in machine / lightburn_mode / starting-point.
    """

    name: str
    settings: HeightmapSettings = Field(default_factory=HeightmapSettings)
    machine: str = "JPT MOPA fiber"
    lightburn_mode: str = "3D Sliced"
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Target-object presets (coin / signet_ring / pendant / plaque / portrait)
# ---------------------------------------------------------------------------

class TargetPresetSummary(BaseModel):
    name: str
    display_name: str
    print_width_mm: float
    print_height_mm: float
    polarity_invert: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Session WebSocket events
# ---------------------------------------------------------------------------

class WsEvent(BaseModel):
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sculptok auto-pull
# ---------------------------------------------------------------------------

class SculptokCreditsResponse(BaseModel):
    """Reflects whether the server has a Sculptok API key configured and,
    if so, the current credit balance."""

    configured: bool
    balance: Optional[int] = None
    cost_pro_2k: int = 15
    cost_pro_4k: int = 30
    cost_normal: int = 10


class SculptokGenerateRequest(BaseModel):
    image_id: str
    style: Literal["normal", "portrait", "sketch", "pro"] = "pro"
    version: Literal["1.0", "1.5"] = "1.5"
    draw_hd: Literal["2k", "4k"] = "2k"
    # Optional heightmap-pipeline settings. When supplied, the route runs
    # the same pre-sculptok conditioning that /render uses (CLAHE,
    # denoise, specular, auto-orient, auto-crop, optional bg-replace) and
    # uploads the prepped photo — so the prep settings actually shape
    # what sculptok sees instead of being cosmetic-only.
    settings: Optional[HeightmapSettings] = None


class SculptokGenerateResponse(BaseModel):
    heightmap_path: str           # server-side filesystem path; passed back via settings.external_heightmap_path
    credits_used: int
    credits_remaining: int
    # Blob id of the photo that was actually uploaded to sculptok (after
    # prep + bg-replace). Lets the wizard show a "Sculptok input" tile
    # so the user can verify what got sent BEFORE the credit was burned.
    sculptok_input_id: Optional[str] = None
    # Blob id of the subject mask, when one was computed during prep.
    # Same mask the wizard would otherwise produce via /mask — exposing
    # it here means the user gets the deliverable for free if they
    # enable subject_mask_enabled or use a bg-replace pattern.
    subject_mask_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

class ApiError(BaseModel):
    code: str
    message: str
    hint: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ApiError
