# Zone-Based Multi-Layer Engraving — Implementation Plan

**Goal**: enable coin-style deep engravings where the device (central figure), field (background), border (decorative band), and rim (outermost ring) are independent heightmap layers with separate LightBurn laser parameters — field fires first at annealing power, device fires last at ablation power, producing the numismatic bright-on-dark contrast effect.

---

## 1. Zone Vocabulary

| Zone | Description | LightBurn priority |
|------|-------------|-------------------|
| **Field** | Smooth background plane behind the device | Fire 1st (anneal, darken) |
| **Device** | Central figure / portrait / subject | Fire last (ablate, raise bright) |
| **Border** | Decorative band between field and rim | Fire 2nd |
| **Rim** | Outermost raised ring | Fire 2nd or 3rd |
| **Exergue** | Bottom text zone below device | Same priority as device |

**Why order matters**: The LightBurn 3D Sliced engine fires layers in the order they appear in the `.lbrn2` file. Field → border → rim → device means the device re-ablates (brightens) material already darkened by the field pass. Reversing the order removes the contrast effect entirely.

---

## 2. How LightBurn Actually Handles Zoning

### 2.1 Outer boundary — vector MaskID (already working)

For coin profiles (`shape: circle` in YAML) the exporter already emits:

```xml
<!-- Tool-layer Ellipse shapes the clip boundary -->
<Shape Type="Ellipse" CutIndex="30" ShapeID="N" UsedByID="0" rx="..." ry="..."/>

<!-- All Bitmap layers reference it -->
<Shape Type="Bitmap" CutIndex="0" ShapeID="0" MaskID="N" SourceFile="pass_00_form.png"/>
```

This is LightBurn's native raster-clip mechanism. It is reliable, lossless, and lets the user drag the mask ring in LightBurn to reposition the coin without touching the PNG.

**Extending to non-circular coins**: replace the Ellipse with a `Path` shape (SVG-style closed path). The `lbrn_writer.py` `ShapeRef` already carries `mask_id`; the writer would only need a `shape_type="Path"` branch with `<VertList>` children. This covers hex tokens, shield shapes, pendants, etc.

### 2.2 Internal zones — raster-only (no vector option)

LightBurn's `MaskID` clips a Bitmap to a vector shape's outer boundary. It cannot clip one 3D Slice Bitmap to a *sub-region* of another Bitmap. There is no "inner MaskID" concept.

**Conclusion**: device vs. field separation is always done by **baking zone masks into the per-pass PNG**. Zone masks are float32 arrays (0=background zone, 1=foreground zone) composited exactly like the existing subject mask:

```python
# Field pass — only fires where device_mask == 0
field_layer = np.clip(1.0 - (1.0 - field_hm) * (1.0 - device_mask), 0.0, 1.0)

# Device pass — only fires where device_mask > 0  
device_layer = np.clip(1.0 - (1.0 - device_hm) * device_mask, 0.0, 1.0)
```

Feathering at zone boundaries = Gaussian blur on the mask before compositing. A 3–10 px sigma eliminates hard seam lines.

### 2.3 Why the existing subject mask step stays useful

The `/mask` route (rembg / BiRefNet / flood-fill) produces the device mask for free. The only change: instead of baking it into the single form pass, it drives two separate CutSetting_Img layers. The wizard mask-drawing step becomes "draw where you want the subject to be" — unchanged from the user's perspective.

---

## 3. Existing Infrastructure to Build On

| Asset | Location | Status |
|-------|----------|--------|
| `background_pattern` field | `apps/api/schemas.py` `HeightmapSettings` | **Done** — 8 patterns including guilloche |
| Pattern generators | `mopa/backgrounds/generators.py` | **Done** — guilloche, stripes, dots, halftone, checkers, solid fills |
| `pre_clean` pass kind | `mopa/stages.py` `PASS_KIND_PRE_CLEAN` | **Exists** — not wired to `user_toggles` in `service_adapter.do_plan()` |
| Ellipse MaskID | `apps/api/service_adapter.py` `add_coin_circle` block | **Done** — tested on silver coins |
| Breakthrough pass | `apps/api/service_adapter.py` do_export_lbrn2 | **Done** — emits `breakthrough.lbrn2` |
| Subject mask baking | `_apply_subject_mask_to_heightmap()` | **Done** |
| Per-pass PNG compositing | `ep.mask * (1 - hm)` loop | **Done** |

---

## 4. The `pre_clean` Wiring Bug

`service_adapter.do_plan()` calls `_plan_passes()` but **never passes `user_toggles`**:

```python
# Current (broken — pre_clean never fires even when pre_clean_enabled=True)
plan = _plan_passes(heightmap=hm, profile=profile, ...)

# Fix
user_toggles = {
    "pre_clean": s.pre_clean_enabled,
    "photo_tonal": s.photo_tonal_enabled,
    "signature": bool(s.signature_text),
}
plan = _plan_passes(heightmap=hm, profile=profile, user_toggles=user_toggles, ...)
```

This is a one-line fix. It should be done before the zone work because zone passes will use the same `user_toggles` mechanism.

---

## 5. Zone Architecture — Data Model

### 5.1 New pass kinds in `stages.py`

```python
PASS_KIND_FIELD     = "zone:field"
PASS_KIND_DEVICE    = "zone:device"
PASS_KIND_BORDER    = "zone:border"
PASS_KIND_RIM       = "zone:rim"
PASS_KIND_EXERGUE   = "zone:exergue"
```

Order in `DEFAULT_PASS_ORDER`:
```
pre_clean → zone:field → zone:border → zone:rim → zone:device → zone:exergue → form → photo_tonal → signature
```

`form` remains the single-layer fallback for non-zoned workflows. When zone passes are enabled, `form` is skipped (or becomes the device pass).

### 5.2 Zone mask sources

| Zone | Mask source |
|------|-------------|
| `zone:device` | User-drawn subject mask (the existing `/mask` route) |
| `zone:field` | `1 - device_mask` — the logical inverse |
| `zone:border` | Radial annulus: `coin_radius - border_width_px < r <= coin_radius` |
| `zone:rim` | Radial annulus: outermost N px of the coin circle |
| `zone:exergue` | Rectangle in the lower quarter of the coin area |

Border and rim masks are computed analytically from the coin geometry — no user drawing required.

### 5.3 Zone heightmaps

Each zone can use:
1. **The same heightmap** (sculptok output), composited through the zone mask — simplest.
2. **A procedural heightmap** (pattern from `mopa/backgrounds/generators.py`) — for field guilloché.
3. **An external heightmap** per zone (future: upload a separate sculptok run for the border).

For the Hecate coin workflow the user already does externally: ChatGPT generates a heightmap that contains the border pattern baked in. The zone system makes this native.

### 5.4 Zone heightmap for the field

```
field_hm = alpha * background_pattern + (1 - alpha) * flat_surface
```

Where `alpha` = `background_intensity` (existing field) and `flat_surface` = `background_value` (existing field). This is exactly what the current `background_pattern` pipeline does — it already composites the pattern into the heightmap. The only new thing is emitting it as a **separate LightBurn layer** with field-appropriate laser parameters.

---

## 6. Profile YAML Extension for Zone Parameters

```yaml
# Existing (unchanged)
lightburn_starting_point:
  speed_mm_s: 600
  power_percent: 75
  ...

# New: per-zone laser override
zone_params:
  field:
    speed_mm_s: 800      # faster, lower power — annealing, not ablation
    power_percent: 55
    frequency_khz: 80
    passes: 2            # two annealing sweeps for uniform darkening
  device:
    speed_mm_s: 600
    power_percent: 75
    frequency_khz: 60    # primary sculptok depth pass
  border:
    speed_mm_s: 700
    power_percent: 60
    frequency_khz: 70
  rim:
    speed_mm_s: 900
    power_percent: 40    # shallow raised ring — low power
    frequency_khz: 100

# Optional zone geometry overrides (defaults derived from print_width/height)
zone_geometry:
  border_width_mm: 1.5
  rim_width_mm: 0.5
  exergue_height_fraction: 0.18
```

The `zone_params` values are lifted by `service_adapter` into `ColorEntry.raw` overrides, following the same pattern as the existing `kind_color_overrides` and breakthrough pass.

---

## 7. LightBurn `.lbrn2` Output Structure

For a fully zoned coin the zip would contain:

```
project.lbrn2          ← multi-layer 3D Slice project
pass_00_zone_field.png ← field heightmap (guilloche pattern composited)
pass_01_zone_border.png
pass_02_zone_rim.png
pass_03_zone_device.png← sculptok subject through device mask
breakthrough.lbrn2     ← separate 1-pass spray-bond file (reflective metals)
breakthrough.png
subject_mask.png       ← deliverable for LightBurn Trace workflow
```

Inside `project.lbrn2`:

```xml
<CutSetting type="Scan"> <!-- Field: index 1, annealing params --> </CutSetting>
<CutSetting type="Scan"> <!-- Border: index 2 --> </CutSetting>
<CutSetting type="Scan"> <!-- Rim: index 3 --> </CutSetting>
<CutSetting type="Scan"> <!-- Device: index 4, ablation params --> </CutSetting>

<!-- Bitmaps in fire order (field first, device last) -->
<Shape Type="Bitmap" CutIndex="1" ShapeID="0" MaskID="4" SourceFile="pass_00_zone_field.png"/>
<Shape Type="Bitmap" CutIndex="2" ShapeID="1" MaskID="4" SourceFile="pass_01_zone_border.png"/>
<Shape Type="Bitmap" CutIndex="3" ShapeID="2" MaskID="4" SourceFile="pass_02_zone_rim.png"/>
<Shape Type="Bitmap" CutIndex="4" ShapeID="3" MaskID="4" SourceFile="pass_03_zone_device.png"/>

<!-- Tool-layer Ellipse shared by all bitmaps via MaskID=4 -->
<Shape Type="Ellipse" CutIndex="30" ShapeID="4" UsedByID="0" rx="..." ry="..."/>
```

**Multi-MaskID note**: LightBurn allows only one `MaskID` per Bitmap shape. All bitmaps share the coin Ellipse. Internal zone separation is done entirely in the PNG content, not by MaskID.

---

## 8. Implementation Phases

### Phase 0 — Fix the `pre_clean` wiring bug *(1 hour)*

Wire `pre_clean_enabled` and other toggles into `service_adapter.do_plan()`. Unblocks pre-clean pass for all users immediately.

**Files**: `apps/api/service_adapter.py` ~line 365 (the `do_plan` call site).

### Phase 1 — Zone heightmap compositing *(1 day)*

Add `zone_hm_for_pass()` function in `mopa/zones.py`:
- Takes `heightmap`, `mask`, `background_pattern`, `background_*` settings
- Returns a composited float32 heightmap for one zone
- Handles feathering at boundaries

Add `PASS_KIND_FIELD`, `PASS_KIND_DEVICE`, etc. to `stages.py`.

Add `zone_masks_from_geometry()` function:
- Computes border/rim/exergue masks analytically from coin radius
- Applies Gaussian feathering (3–5 px sigma at zone edges)

**Files**: `mopa/zones.py` (new), `mopa/stages.py`.

### Phase 2 — Profile YAML zone params *(half day)*

Parse `zone_params` in `service_adapter._profile_kind_color_overrides()`.  
Wire zone `ColorEntry` rows in `service_adapter.do_plan()`.  
Each zone pass gets its `CutSetting` index from the next available slot after the existing passes.

**Files**: `apps/api/service_adapter.py`, `mopa/profiles.py`.

### Phase 3 — Multi-layer `.lbrn2` export *(1 day)*

In `service_adapter.do_export_lbrn2()`:
- Replace the single-pass loop with zone-aware compositing
- Maintain fire order: field → border → rim → device
- All bitmaps share the coin Ellipse `MaskID` (existing mechanism)
- Emit one PNG per zone

**Files**: `apps/api/service_adapter.py`.

### Phase 4 — Wizard UI zone controls *(1–2 days)*

Add zone panel in wizard after profile selection:
- Toggle per zone (enabled/disabled)
- Background pattern picker for field (reuse existing `background_pattern` UI)
- Depth slider per zone (relative to profile starting point)
- Preview: color-coded zone overlay on the coin preview image

**Files**: `apps/web/src/app/features/wizard/wizard-shell.component.ts` and related.

### Phase 5 — Style preset system *(2 days)*

A style preset is a named bundle of:
- `field_pattern` (one of the existing generators + parameters)
- `border_asset` (from a library of pre-generated border heightmap tiles)
- `rim_params` (depth, width)
- `device_depth_boost` (multiplier on base profile depth)

Store presets as YAML in `mopa/style_presets/`. The wizard exposes a gallery picker — one card per preset with a thumbnail render.

**Files**: `mopa/style_presets/` (new dir), `apps/api/routes/` (GET /style-presets), wizard component.

### Phase 6 — Border/rim asset library *(ongoing)*

Generate a set of 512×512 px tileable heightmap tiles for:
- Beaded rim (the Hecate coin used this)
- Acanthus scroll border
- Laurel wreath
- Cable twist
- Key (meander) pattern
- Rope twist

Store as 16-bit PNGs in `mopa/assets/borders/`. The zone compositing code tiles or stretches them along the annular region. This is a creative/content task, not an architecture task.

---

## 9. What to Reconsider or Ditch

### Subject mask wizard step — keep, repurpose

The mask drawing step is still needed but its role shifts: it no longer drives a single-pass bake, it drives the `zone:device` mask. The UX message changes from "mask the subject" to "draw where the central design is." For coins with a clear subject (portrait, deity), rembg auto-mask will cover most cases without user intervention.

### `background_pattern` as a field property

The current `background_pattern` field in `HeightmapSettings` composites the pattern **before** sculptok sees the photo (replacing the background pixels in the *input image*). For zone architecture, the field pattern should composite **after** sculptok into a separate layer. Both modes are valid for different workflows:

- **Before sculptok** (current): sculptok sees the pattern as texture and may sculpt depth into it. Good for seamless integration.
- **After sculptok** (zone): pattern fires at independent laser parameters. Good for bright-on-dark contrast. 

Consider renaming the current setting to `input_background_pattern` (or adding an `apply_at` option) to avoid ambiguity.

### Color k-means passes

The `color:*` pass kind (LAB k-means clusters from the photo) is largely superseded by zone-based design. Consider deprecating or hiding this behind a developer flag once zones are working — it adds complexity without clear user benefit for the numismatic use case.

---

## 10. Near-Term Quick Wins (before Phase 1)

These require minimal code and have immediate user impact:

1. **Fix `pre_clean` wiring** — already described in §4.
2. **Expose `background_pattern` in wizard UI** — the API field exists; the wizard just needs a picker. Gives users guilloche/stripes/dots on their coin field today.
3. **Expose `polarity_invert` in wizard** — toggle for signet ring / recessed designs. Already a schema field.
4. **Show `conditioned_id` in wizard** — the render response already returns it. Lets users see "what sculptok actually saw" before burning a credit.
