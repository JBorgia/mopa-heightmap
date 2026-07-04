# Zone-Based Multi-Layer Engraving — Implementation Plan

**Goal**: enable coin-style deep engravings where the device (central figure), field (background), border (decorative band), and rim (outermost ring) are independent heightmap layers with separate LightBurn laser parameters — field fires first at annealing power, device fires last at ablation power, producing the numismatic bright-on-dark contrast effect.

---

## 1. Zone Vocabulary

| Zone | Description | LightBurn priority | Shape-agnostic? |
|------|-------------|-------------------|-----------------|
| **Field** | Smooth background plane behind the device | Fire 1st (anneal, darken) | ✅ Yes — `1 - device_mask` |
| **Device** | Central figure / portrait / subject | Fire last (ablate, raise bright) | ✅ Yes — user-drawn mask |
| **Border** | Decorative band between field and rim | Fire 2nd | ⚠️ Needs polygon-inset for non-circles |
| **Rim** | Outermost raised ring | Fire 2nd or 3rd | ⚠️ Needs polygon-inset for non-circles |
| **Exergue** | Bottom text zone below device | Same priority as device | ✅ Yes — bottom N% of bounding-box height |

**Why order matters**: The LightBurn 3D Sliced engine fires layers in the order they appear in the `.lbrn2` file (community-confirmed behaviour, not officially documented by LightBurn). Field → border → rim → device means the device re-ablates (brightens) material already darkened by the field pass. Reversing the order removes the contrast effect entirely.

**Shape generalisation**: Field, Device, and Exergue zones require no geometry — they are derived from the user-drawn subject mask or bounding-box arithmetic. Border and Rim zones are defined by inward offsets from the blank boundary; for circular blanks this is radial distance, for all other shapes it is a polygon-offset computation (see §5.2).

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

**Extending to non-circular blanks**: replace the Ellipse with a `Path` shape (SVG-style closed path). The `lbrn_writer.py` `ShapeRef` already carries `mask_id`; the writer would only need a `shape_type="Path"` branch with `<VertList>` children. This covers hex tokens, shield shapes, triangles, pendants, etc.

**Donut/annular blanks**: the outer boundary is an Ellipse (or Path); the inner hole is a second inner MaskID boundary. Model as a `Polygon` with a hole (Shapely: `Polygon(outer_ring, [inner_ring])`). The profile exposes `hole_radius_mm`; if set, the exporter emits both an outer and inner shape on the Tool layer. Border and rim masks are computed in the annular region between `hole_radius` and `outer_radius - rim_width`.

### 2.2 Internal zones — raster-only (no vector option)

LightBurn's `MaskID` clips a Bitmap to a vector shape's outer boundary. It cannot clip one 3D Slice Bitmap to a *sub-region* of another Bitmap. There is no "inner MaskID" concept.

**Conclusion**: device vs. field separation is always done by **baking zone masks into the per-pass PNG**. Zone masks are float32 arrays (0=background zone, 1=foreground zone) composited exactly like the existing subject mask:

```python
# Field pass — only fires where device_mask == 0
field_layer = np.clip(1.0 - (1.0 - field_hm) * (1.0 - device_mask), 0.0, 1.0)

# Device pass — only fires where device_mask > 0  
device_layer = np.clip(1.0 - (1.0 - device_hm) * device_mask, 0.0, 1.0)
```

Feathering at zone boundaries = Gaussian blur on the mask before compositing. The correct sigma depends on material:

| Material | Recommended σ | Reason |
|----------|--------------|--------|
| Silver, Brass | 1.5 px | Sharp relief reads well on reflective surfaces |
| Coated steel, dark finishes | 2.5 px | Softer transitions hide laser edge artifacts |
| Anodized aluminum | 3.0 px | Heat-sensitive; spread energy concentration at boundary |

After compositing, apply a bilateral post-filter (`cv2.bilateralFilter`, σ_color=0.10, σ_space=10) to suppress dither-artifact halos at zone seams without smearing the transition itself.

**Critical rule**: never frequency-decompose the sculptok heightmap across multiple layers (e.g., low-frequency field pass + high-frequency device pass from the same depth map). Overlapping pixels at shared frequencies compound depth 3–5× across layers, burning through material. One sculptok PNG per zone; each zone carries its full depth budget independently.

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

| Zone | Mask source | Algorithm |
|------|-------------|-----------|
| `zone:device` | User-drawn subject mask (the existing `/mask` route) | rembg / BiRefNet / flood-fill |
| `zone:field` | Logical inverse of device mask | `1 - device_mask` |
| `zone:border` | Inward offset from blank boundary | **Circle**: radial annulus `radius - border_width_px < r ≤ radius`; **Polygon**: `Shapely.buffer(-border_width_px)` |
| `zone:rim` | Outermost N px of blank boundary | **Circle**: radial annulus; **Polygon**: outer polygon minus `Shapely.buffer(-rim_width_px)` |
| `zone:exergue` | Bottom N% of bounding-box height | `y > (1 - exergue_height_fraction) * bbox_height` — shape-agnostic |

Border and rim masks for **circular blanks** are computed from radial distance. For **all other shapes** (hexagon, triangle, shield, donut), they are computed via Shapely polygon offset (`buffer(-n)`), which produces the correct inward-inset polygon region regardless of shape.

**Donut/annular blanks**: the blank is modelled as `Polygon(outer_ring, [inner_hole_ring])`. The field mask covers the full annular region; border is the outer annular band (`outer_radius - border_width < r ≤ outer_radius`); rim is the innermost band adjacent to the hole (`hole_radius ≤ r < hole_radius + rim_width`). The exergue is suppressed for donut shapes (no bottom text zone makes sense on a ring).

All masks receive Gaussian feathering before compositing. `sigma = base_sigma * profile.zone_boundary_sigma_scale` where `base_sigma = 1.5 px`.

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
# Existing shape field — extended enum
shape: circle          # circle | rectangle | hexagon | triangle | donut | shield | path

# Existing (unchanged)
lightburn_starting_point:
  speed_mm_s: 600
  power_percent: 75
  ...

# New: per-zone laser override (example: 60W MOPA silver calibrated values)
zone_params:
  field:
    speed_mm_s: 2100     # fast + low power = surface annealing, not ablation
    power_percent: 32
    frequency_khz: 100
    pulse_width_ns: 185
    passes: 2            # two annealing sweeps for uniform field darkening
  device:
    speed_mm_s: 600      # same as lightburn_starting_point — full sculptok depth
    power_percent: 75
    frequency_khz: 60
    pulse_width_ns: 130  # short pulse: explosive vaporisation, prevents pooling on silver
  border:
    speed_mm_s: 800
    power_percent: 50
    frequency_khz: 80
    pulse_width_ns: 160
  rim:
    speed_mm_s: 1000
    power_percent: 38    # shallow raised ring — low power, fast
    frequency_khz: 100
    pulse_width_ns: 185

# Zone geometry — defaults derived from print_width/height; override here
zone_geometry:
  border_width_mm: 1.5
  rim_width_mm: 0.5
  exergue_height_fraction: 0.18   # bottom 18% of bounding-box height (shape-agnostic)
  hole_radius_mm: ~               # donut inner hole radius (null = no hole)

# Zone boundary feathering scale factor (multiplies base sigma of 1.5 px)
# silver/brass = 1.0, coated steel = 1.7, anodized aluminum = 2.0
zone_boundary_sigma_scale: 1.0
```

The `zone_params` values are lifted by `service_adapter` into `ColorEntry.raw` overrides, following the same pattern as the existing `kind_color_overrides` and breakthrough pass. The `shape` field drives which mask algorithm `zones.py` uses for border/rim: radial distance for `circle`; Shapely `buffer(-offset_px)` for all polygon shapes; dual-boundary Shapely polygon-with-hole for `donut`.

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
- Applies `cv2.bilateralFilter(σ_color=0.10, σ_space=10)` post-compositing to suppress dither-artifact halos at zone seams

Add `PASS_KIND_FIELD`, `PASS_KIND_DEVICE`, etc. to `stages.py`.

Add `zone_masks_from_geometry()` function:
- **Circle blanks**: border/rim as radial annulus (`coin_radius - width_px < r ≤ coin_radius`)
- **All other shapes** (hexagon, triangle, shield): border/rim via `shapely.Polygon.buffer(-offset_px)` — polygon inset
- **Donut**: outer Shapely polygon with inner hole (`Polygon(outer, [inner])`); rim is band adjacent to inner hole
- Exergue: `y > (1 - exergue_height_fraction) * bbox_height` — shape-agnostic, no geometry library needed
- Feathering sigma: `base_sigma=1.5 * profile.zone_boundary_sigma_scale` (profile field, default 1.0)

Add `shape` enum to `_KNOWN_TOP_LEVEL_KEYS` in `profiles.py`: `circle | rectangle | hexagon | triangle | donut | shield | path`.

**Files**: `mopa/zones.py` (new), `mopa/stages.py`, `mopa/profiles.py`.

### Phase 2 — Profile YAML zone params *(half day)*

Parse `zone_params` in `service_adapter._profile_kind_color_overrides()`.  
Wire zone `ColorEntry` rows in `service_adapter.do_plan()`.  
Each zone pass gets its `CutSetting` index from the next available slot after the existing passes.

Add `zone_geometry` and `zone_boundary_sigma_scale` to `_KNOWN_TOP_LEVEL_KEYS` in `profiles.py`.

Seed `zone_params` in `mopa_60w_silver.yaml` with the calibrated field-zone annealing values from §6. This is the reference implementation that other profiles will adapt.

**Files**: `apps/api/service_adapter.py`, `mopa/profiles.py`, `profiles/mopa_60w_silver.yaml`.

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

## 9. Shape Support Matrix

| Shape | Boundary algorithm | Border/Rim mask | Exergue | Status |
|-------|-------------------|-----------------|---------|--------|
| Circle / Oval | Ellipse | Radial annulus | Bottom 18% | ✅ Done |
| Rectangle | Rect clip | `buffer(-n)` inset rect | Bottom 18% | ✅ Profile exists |
| Hexagon | 6-vertex Path + `buffer(-n)` | Polygon inset | Bottom 18% | ⚠️ Phase 1 |
| Triangle | 3-vertex Path + `buffer(-n)` | Polygon inset | Suppress (too small) | ⚠️ Phase 1 |
| Donut / Annular | Ellipse + inner hole Ellipse | Annular bands | Suppress | ⚠️ Phase 1 |
| Shield / Custom | SVG Path + `buffer(-n)` | Polygon inset | Bottom 18% | ⚠️ Phase 1 |
| Signet Ring (flat) | Rectangle, `polarity_invert` | Optional border | Bottom 15% | ✅ Preset exists |

Rotary-axis (curved shank) engraving is **explicitly out of scope** — the system assumes flat blanks throughout.

---

## 10. What to Reconsider or Ditch

### Subject mask wizard step — keep, repurpose

The mask drawing step is still needed but its role shifts: it no longer drives a single-pass bake, it drives the `zone:device` mask. The UX message changes from "mask the subject" to "draw where the central design is." For coins with a clear subject (portrait, deity), rembg auto-mask will cover most cases without user intervention.

### `background_pattern` as a field property

The current `background_pattern` field in `HeightmapSettings` composites the pattern **before** sculptok sees the photo (replacing the background pixels in the *input image*). For zone architecture, the field pattern should composite **after** sculptok into a separate layer. Both modes are valid for different workflows:

- **Before sculptok** (current): sculptok sees the pattern as texture and may sculpt depth into it. Good for seamless integration.
- **After sculptok** (zone): pattern fires at independent laser parameters. Good for bright-on-dark contrast. 

Consider renaming the current setting to `input_background_pattern` (or adding an `apply_at` option) to avoid ambiguity.

### Heightmap frequency decomposition — forbidden

Never split a single sculptok heightmap into frequency bands across multiple layers (low-frequency field pass + high-frequency device pass from one depth map). Overlapping pixels at shared frequencies compound depth **3–5× across layers**, burning through material. Each zone must carry an independent depth budget. This rule should be enforced as a comment at the zone compositing call site in `do_export_lbrn2()`.

### Color k-means passes

The `color:*` pass kind (LAB k-means clusters from the photo) is largely superseded by zone-based design. Consider deprecating or hiding this behind a developer flag once zones are working — it adds complexity without clear user benefit for the numismatic use case.

---

## 11. Near-Term Quick Wins (before Phase 1)

These require minimal code and have immediate user impact:

1. **Fix `pre_clean` wiring** — already described in §4.
2. **Expose `background_pattern` in wizard UI** — the API field exists; the wizard just needs a picker. Gives users guilloche/stripes/dots on their coin field today.
3. **Expose `polarity_invert` in wizard** — toggle for signet ring / recessed designs. Already a schema field.
4. **Show `conditioned_id` in wizard** — the render response already returns it. Lets users see "what sculptok actually saw" before burning a credit.
