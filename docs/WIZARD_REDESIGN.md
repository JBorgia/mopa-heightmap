# MOPA Heightmap Studio — Wizard Redesign

> **Status: Canonical design, July 2026.**
> Supersedes: `IMPLEMENTATION_PLAN.md` (deleted), `docs/PLAN.md` (deleted), `docs/SONNET_UI_MIGRATION_BRIEF.md` (deleted).

---

## Design Philosophy

### Credits-last architecture

The most important constraint shaping this flow: **Sculptok costs credits; everything else is free.** This forces a natural ordering:

1. Every decision that does not require the depth AI — shape, size, composition, image pre-processing, overlay text — happens *before* any API call.
2. The depth map is generated exactly once per creative intent. The user refines around it, not through it.
3. Material selection and laser parameters happen *after* rendering, so the same depth render can be exported to multiple materials without re-generating.

### Operator mental model

The target user is a laser engraver operator, not a graphic designer or AI practitioner. They think in terms of:

- "What am I making?" (blank type + shape)
- "What image goes on it?" (photo + positioning)
- "How do I want the depth to look?" (relief style)
- "What material am I running today?" (laser params)

The wizard mirrors this sequence exactly.

### Canvas-as-workspace

The canvas is not a preview pane — it is the primary workspace. Every spatial decision (crop position, scale, overlay placement, zone boundaries) is made by direct manipulation on the canvas, not through numeric sliders. Sliders are secondary controls for fine-tuning what the canvas establishes.

---

## The 8-Step / 4-Stage Flow

```
STAGE 1: COMPOSE (free — no API calls)
  Step 1 │ Blank Setup        — shape + project type + physical size
  Step 2 │ Photo & Canvas     — upload + crop/reposition + background
  Step 3 │ Image Prep         — pre-processing before depth AI
  Step 4 │ Checkpoint         — local depth preview + go/no-go before credits

STAGE 2: GENERATE (costs credits — one API call)
  Step 5 │ Depth Map          — Sculptok run + enhancement controls

STAGE 3: REFINE (free — no API calls)
  Step 6 │ Subject & Zones    — mask refinement + zone boundary control
  Step 7 │ Output Simulation  — lit 3D render preview of finished piece

STAGE 4: OUTPUT (free — no API calls)
  Step 8 │ Material & Export  — laser params + multi-format export
```

---

## Stage 1: Compose

### Step 1 — Blank Setup

**What:** Define the physical object being engraved.

**Why first:** Blank shape and size constrain everything downstream — the crop aspect ratio, the zone geometry, the pixel-per-mm resolution, and which material profiles appear. Getting this wrong means all subsequent work is invalidated.

**Controls:**

| Group | Control | Notes |
|---|---|---|
| **Shape** | Shape chip grid | See shape list below |
| **Project type** | Type chip grid | See project type list below |
| **Physical size** | Width × Height (mm) | Drives px/mm at render resolution |
| **Hole (donut)** | Inner radius (mm) | Shown only when shape = donut |
| **Orientation** | Portrait / Landscape toggle | Swaps W×H for non-circular shapes |

**Shapes supported:**

| Shape | Notes |
|---|---|
| Circle | Standard coin, medallion, button |
| Rectangle | Plaque, knife scale, dog tag |
| Hexagon | Challenge coin variant, hex tile |
| Triangle | Pendant, arrowhead |
| Donut / Annular ring | Washer, ring face, annular coin |
| Shield | Heraldic, badge, lapel pin |
| Oval / Ellipse | Brooch, cameo |
| Star | 5- or 6-point; point count selector appears |
| Freeform (SVG path) | User uploads a closed SVG path; used as mask |

**Project types** (drives default material profile, zone widths, depth budget):

| Type | Shape hints | Default zones |
|---|---|---|
| Coin — collector | Circle, hexagon | rim + border + field + device |
| Coin — challenge | Circle | rim + field + device |
| Signet ring face | Circle, oval, shield | rim + device (no border) |
| Pendant | Any | rim + field + device |
| Plaque | Rectangle | rim + border + field + device + exergue |
| Dog tag | Rectangle | rim + field + device |
| Knife scale | Rectangle (tall) | field + device |
| Watch case back | Circle | rim + field + device |
| Portrait / wall art | Rectangle | field + device |
| Custom | Any | User configures zones manually |

---

### Step 2 — Photo & Canvas

**What:** Upload the source photo, position and crop it over the blank shape, set background treatment.

**Why here:** Composition happens before any image processing or depth generation. Moving the subject in the crop changes what pixels Sculptok sees, which changes the depth output. This must be locked before pre-processing or generation.

**Canvas interaction:**
- The blank shape is rendered as a mask overlay on a 1:1 canvas.
- The photo sits underneath, draggable and scalable.
- Resize handles on the photo allow free scale (locked aspect).
- The crop/visible region is always the intersection of the photo and the blank shape outline.
- Crop center `(cx, cy)` in image-fraction coordinates is sent to the backend so auto-crop aligns to the user's intent, overriding face-detection when the user has positioned manually.

**Background treatment** (what fills pixels outside the subject but inside the blank):

| Option | Effect on depth map |
|---|---|
| Flat (zero depth) | Background = white (no engraving) |
| Flat (mid depth) | Background = 50% gray |
| Gradient (radial) | Falls off from subject edge outward |
| Match border | Background tone matches the chosen border zone |
| Transparent (alpha) | Requires subject mask; fills after masking step |

**Overlays** (compositional elements added before depth generation):

- **Text overlay:** Engraver's text (mint mark, motto, year, name). Font, size, position, baseline arc. Rendered into the image as black-on-white pixels before Sculptok, so depth AI reads them as relief elements.
- **Emblem / logo:** Upload a second PNG (e.g., hallmark, logo); composited at a chosen position and scale.
- **Reference grid:** Faint registration marks to help align physical blank to engraving — not engraved, used only for positioning guidance.

**Reference image calibration:** Upload a previously engraved piece photo to calibrate the depth → gray-level response curve for this machine + material. (Stores a per-machine correction LUT; surfaced in Step 8.)

---

### Step 3 — Image Prep

**What:** Pre-processing applied to the photo before Sculptok ingests it. These controls manipulate the photograph, not the depth map.

**Why before generation:** Sculptok (and any monocular depth net) reads the image literally. Specularity confuses it into treating bright highlights as raised geometry. Poor contrast compresses the depth range. These corrections improve depth quality without costing credits.

**Controls:**

| Control | Purpose |
|---|---|
| Specular removal | Desaturates and normalises bright specular highlights; auto-enabled for metal project types |
| CLAHE contrast | Adaptive histogram equalisation (uint16 precision); lifts local contrast in shadow regions |
| White balance | Warm/cool slider; corrects color-cast from photography lighting |
| Denoising | Mild bilateral denoise; reduces depth noise from image grain |
| Vignette correction | Lift corners; counteracts lens darkening that Sculptok may read as depth |
| Micro-contrast | Unsharp mask on the input; helps Sculptok resolve fine surface texture |
| Depth style preset | Preloaded parameter bundle (portrait / standard / coin / custom) |

**Depth style presets** (apply after generation, but chosen here so the user commits before spending credits):

| Preset | Post-processing applied |
|---|---|
| Portrait | Gentle percentile stretch (5–95 %) + gamma 1.2 |
| Standard | Moderate stretch (3–97 %) + gamma 1.35 + light unsharp |
| Coin | Aggressive stretch (2–98 %) + gamma 1.5 + CLAHE (uint16) + unsharp |
| Off | Passthrough — raw Sculptok output |

---

### Step 4 — Checkpoint

**What:** A local depth-preview computed from the photo without calling Sculptok, plus a pre-flight checklist that must be green before the "Generate" button is active.

**Why:** Gives the operator a sense check and catches obvious problems (wrong crop aspect, missing subject, poor contrast) before credits are spent.

**Local preview:** A classical (non-AI) depth estimate derived from luminance + edge energy. Fast (< 1 s), purely local, no API call. Shown as a grayscale relief preview and an isometric 3D thumbnail. **Not** the actual depth map — clearly labelled "preview estimate, not the final render."

**Pre-flight checklist** (green = ready to generate):

| Check | Pass condition |
|---|---|
| Blank size set | Width and height > 0 |
| Photo loaded | Source image present |
| Crop positioned | Crop area covers ≥ 40 % of the blank |
| Resolution adequate | px/mm ≥ 10 at chosen size |
| Specular check | No overexposed regions > 5 % of subject area (warning, not block) |
| Credits available | Remaining credits > 1 |

**Cost estimate:** Shows credits required for this job before the user commits.

---

## Stage 2: Generate

### Step 5 — Depth Map

**What:** Run Sculptok on the prepared image; display the result; apply enhancement controls.

**Why exactly here:** All free composition and preparation work is locked. The user has seen the pre-flight estimate and committed to spending credits. This is the only step that costs.

**Flow:**
1. Progress strip: preparing → uploading → generating → packaging → done / error.
2. On completion: show heightmap in split-view (original photo | depth map) with the before/after slider.
3. Enhancement controls appear inline (not a separate step) — they re-run only the post-processing substage, not Sculptok:
   - Depth style (portrait / standard / coin / off) — selected in Step 3, adjustable here.
   - Polarity — light-raises vs dark-raises toggle.
   - Auto-stretch range — percentile lo/hi sliders.
   - Gamma — power-law curve.
4. "Re-generate" button available — reruns Sculptok with the same prepared image. Allowed at any time. Previous depth result is replaced.
5. On success: wizard advances to Stage 3. Back navigation to Stage 1 remains available.

---

## Stage 3: Refine

### Step 6 — Subject & Zones

**What:** Optional mask refinement and zone boundary control. Both are editable post-generation with no API cost.

**Subject mask:**
- Auto-mask runs immediately after depth generation (BiRefNet backend).
- Canvas shows the mask as a colour overlay (red = background, green = subject).
- Brush tools: add / remove / smooth. Lasso select. Edge refinement slider.
- Mask determines which pixels participate in the `zone:device` pass and which pixels are background.

**Zone boundaries:**
- Visual overlay on canvas: rim band (outermost), border band, field, exergue strip (bottom text area).
- Sliders: rim width (mm), border width (mm), exergue height fraction.
- Sigma scale: controls how much the zone boundaries feather (blend rather than hard cut).
- Live preview: zone overlay updates as sliders move; no recompute required.

**Donut / annular controls** (shown when shape = donut):
- Inner hole radius (mm).
- Inner rim band at hole edge: on/off.

---

### Step 7 — Output Simulation

**What:** Render the finished engraving as a lit 3D surface to let the operator visualise the result before committing to a job.

**Why before export:** Catches depth-range problems (too flat, too deep, inverted) that are invisible in a grayscale preview but obvious in an isometric or angled 3D view. Saves material waste.

**Views:**
- Isometric 3D: height-mapped mesh with directional lighting, rotatable.
- Cross-section: 1D depth profile along a user-drawn line on the canvas.
- Zone comparison: side-by-side of each zone's depth layer (field, border, rim, device).

**Lighting presets:**
- Raking light (reveals fine texture).
- Overhead (uniform depth reading).
- Metal finish simulation (specular highlight approximation for brass / steel / silver / gold).

**Calibration correction:** If a machine calibration LUT was loaded in Step 2, it is applied here so the simulation reflects actual material response rather than theoretical depth.

---

## Stage 4: Output

### Step 8 — Material & Export

**What:** Choose laser parameters, preview the pass stack, and export. This step is intentionally designed for re-entry — the user can return here with the same depth render and export for a different material without touching any earlier step.

**Material profile:**
- Chip selector: brass / stainless / aluminium / copper / silver / titanium / custom.
- Profile drives: power %, speed mm/s, frequency kHz, pulse width ns, pass count, line interval mm, zone-specific overrides.
- Zone-specific controls appear as an expandable table with per-zone overrides for power, speed, and passes.

**Pass stack:**
- Visual list of passes in fire order with labels and color-slot indicators.
- Toggle switches: enable/disable individual passes.
- Cleanup pass toggle (debris clearing after device zone).
- Preview: estimated job time based on physical size, line interval, and pass count.

**Export targets:**

| Format | Contents | Use |
|---|---|---|
| `.lbrn2` | Full LightBurn project: all passes, zone bitmaps, laser params | Primary laser job file |
| PNG (heightmap) | 8-bit grayscale, white-raised | Manual import into LightBurn |
| PNG (zone masks) | Per-zone PNG set (field, border, rim, device, exergue) | Custom LightBurn setup |
| STL | Mesh from heightmap (for 3D preview / CNC) | Not for laser use |
| Settings JSON | Full parameter snapshot, machine-readable | Reproducibility sidecar |

**Re-export path:** After any export, the user can change the material profile and export again without returning to earlier steps. The depth render is cached in session.

**Calibration ramp export:** Generates a gray-ramp test PNG (8 bands, 0–255) matched to the current material profile for burning before production work.

---

## What Lives Where (Implementation Notes)

### Pre-API vs post-API boundary

```
Pre-API (Steps 1–4):          Post-API (Steps 6–8):
  input_remove_specular          zone masks (zones.py)
  input_clahe                    subject mask (masker)
  input_white_balance            zone_hm_for_pass()
  input_auto_crop_cx/cy          heightmap_enhance_mode
  input_auto_crop_aspect         pass stack (stages.py)
  text/emblem overlay            lbrn2 export
  background fill
```

### No frequency decomposition across zone layers

Zone layers each receive the **full heightmap**, masked to their region. They do **not** receive decomposed frequency bands. Overlapping frequency bands from the same heightmap across zone layers would compound depth 3–5× and burn through material.

### Crop center hint

User canvas position `(cx, cy)` is forwarded as `input_auto_crop_cx` / `input_auto_crop_cy` (image-fraction coordinates, [0, 1]). When these differ from 0.5 by > 1e-4, `auto_crop_to_aspect()` treats the user's position as authoritative and skips face/saliency detection.

### Zone color-slot assignments

```python
ZONE_INDEX = {
    "zone:field":   2,   # C02
    "zone:border":  3,   # C03
    "zone:rim":     4,   # C04
    "zone:device":  1,   # C01  — replaces form pass
    "zone:exergue": 5,   # C05
}
```

`zone:device` reuses C01 (same slot as the form pass) so it replaces the single-pass depth carve seamlessly when zones are enabled.

---

## Shapes: Detailed Notes

| Shape | Geometry | Zone mask strategy |
|---|---|---|
| Circle | Radial distance from center | Concentric annuli |
| Donut | Two concentric circles; inner hole blanked | Rim at both outer and inner edge |
| Rectangle | Chebyshev distance (axis-aligned box) | Inset bands on all four sides |
| Hexagon | Shapely polygon buffer(-inset_px) | Polygon inset for each zone ring |
| Triangle | Same as hexagon, 3 vertices | Same |
| Shield | Falls back to rectangle geometry | Same as rectangle |
| Oval | Ellipse: scaled radial distance | Scaled annuli |
| Star | N-point star polygon; Shapely | Polygon inset |
| Freeform SVG | SVG path → Shapely polygon | Polygon inset |

---

## Project Type Notes

| Type | Typical size range | Zone defaults | Depth budget |
|---|---|---|---|
| Coin — collector | 25–40 mm dia | Full 5-zone | High (256 passes) |
| Coin — challenge | 38–50 mm dia | rim + field + device | Medium |
| Signet ring face | 10–20 mm dia | rim + device | High (tight px/mm) |
| Pendant | 15–35 mm | rim + field + device | Medium |
| Plaque | 50–150 mm | Full 5-zone + large exergue | Medium–low |
| Dog tag | 27×50 mm std | rim + field + device | Medium |
| Knife scale | 25×130 mm typ | field + device | Medium |
| Watch case back | 30–45 mm dia | rim + field + device | Low (shallow relief) |
| Portrait / wall art | 50–200 mm | field + device | Low–medium |
| Custom | Any | User defined | User defined |

---

## Decisions Recorded Here

- **Credits-last**: material profile selection moved to Step 8 so one render → many exports. Never require a re-generate to switch materials.
- **Canvas is workspace**: drag-to-position crop instead of numeric entry; canvas cx/cy wired to backend.
- **Text overlay pre-Sculptok**: text composited into the image before depth generation so Sculptok reads letters as physical geometry, not post-processing labels.
- **Local depth estimate at checkpoint**: fast luminance-based preview before API call; clearly labelled as estimate, not output.
- **Physical simulation before export**: isometric 3D render with material-appropriate lighting lets operator catch depth inversions without wasting material.
- **Re-export as first-class path**: Step 8 is re-enterable; cached depth render survives material profile change.
- **Freeform SVG shape**: included because customers with custom die shapes (logos, crests) need arbitrary blank outlines, not just primitives.
- **Star shape**: 5- and 6-point police/military badges are common in the signet/challenge coin market.
- **Oval/ellipse**: brooch and cameo market uses this geometry frequently.
- **Reference image calibration**: machine-to-machine variation in MOPA response curves is significant; a per-machine LUT loaded in Step 2 and applied in Step 7 simulation prevents depth-range surprises at burn time.
