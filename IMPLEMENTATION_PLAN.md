# MOPA Heightmap Studio — Implementation Plan

> Status: **Living document.** Last updated during the multi-stage AI architecture pivot.
> Goal: photo → MOPA-engraver-ready heightmap with skull-medallion-class detail, with as much of the configuration auto-tuned as possible.

---

## 1. Guiding principles

1. **AI does the heavy lifting, the user does the tasting.** Every stage runs with a smart default produced by an AI step (mask, depth, normals, region weights, exposure, …) and exposes its single most-important knob to a live-preview slider. The user never *has* to think; they always *can* override.
2. **Hybrid > monolithic.** No single model produces engraving-quality output. We compose: classical geometry (Frankot-Chellappa), modern depth nets (Depth-Anything-V2), normals (DSINE/Marigold), segmentation (BiRefNet/Sapiens), generative augmentation (ControlNet relief). Each layer is independently testable and independently disable-able.
3. **Pass stack, not pipeline.** Engraving = an *ordered, opt-in* stack of laser passes (Form, Cleanup, Detail, Shading, Polish, Color₁…Color_N, Signature). Disable color → the planner just omits those layers. Add a custom material profile → the planner regenerates the stack. The .lbrn project is the materialised stack.
4. **Live preview is a first-class feature.** Every knob change re-runs only the affected (cached) substage and re-renders a 512² preview in < 200 ms. Heavy stages (Marigold, Hunyuan3D) are gated behind explicit "Recompute heavy" buttons.
5. **Color profiles are data, not code.** The user-supplied LightBurn color cards (`Colour20W-M7.lbrn2` … `Colour100W-M7.lbrn2`) are the canonical source of truth for MOPA color stages. We *ingest* them at startup; we never hand-author MOPA tables.

## 2. Why we are rebuilding

The current Stage B "detail injection" blends photo luminance / high-pass into the depth map. This is **not physical** — color/albedo is not depth, so engravings come out looking like wet ink rather than carved relief. The reference quality the user asked for ("skull medallion") requires geometry-derived detail, not pixel-shading borrowed from the photo.

The TKOSEI / ReliefGenerater HuggingFace Space (which produces the kind of output we want) confirms three things we are missing:

1. **A modern depth backbone** — they use **Depth-Anything-V2-Large**, not ZoeDepth.
2. **Hard subject masking** — background is forced to a known plane via `np.where(depth == coef_far, …)`; no detail is injected there.
3. **A *very small* photo-luminance contribution** — α ≈ 0.05, not the 0.4 we are currently shipping.

We also discovered a richer technique that ReliefGenerater does **not** use: **surface-normal estimation + Frankot-Chellappa integration**. This recovers genuine micro-relief (pores, fabric weave, hair strands) from the photo as actual height information, with no hallucination.

---

## 3. Target architecture (image processing)

```
                     ┌─────────────────────────────────┐
                     │ 0. Preprocess (optional)        │
                     │   - Auto-rotate/crop            │
                     │   - CLAHE / exposure normalize  │
                     │   - Real-ESRGAN if < 1024 px    │
                     │   - Marigold-IID delight (opt)  │
                     └────────────────┬────────────────┘
                                      ▼
        ┌────────────────────────┐         ┌──────────────────────────┐
        │ 1. Subject mask        │         │ 2. Bulk depth            │
        │   BiRefNet (default)   │         │   Depth-Anything-V2-Base │
        │   rembg / RMBG-2 (alt) │         │   (Large opt, NC-only)   │
        └───────────┬────────────┘         └─────────────┬────────────┘
                    │                                    │
                    │   ┌────────────────────────────────┘
                    │   │
                    ▼   ▼
        ┌─────────────────────────┐         ┌──────────────────────────┐
        │ 3. Surface normals      │         │ 4. Optional: body parts  │
        │   DSINE (default)       │         │   Sapiens-Seg (portraits)│
        │   Marigold-Normals (opt)│         │   gives per-region weights│
        └───────────┬─────────────┘         └─────────────┬────────────┘
                    │                                     │
                    ▼                                     │
        ┌─────────────────────────┐                       │
        │ 5. Frankot-Chellappa    │                       │
        │   FFT integration:      │                       │
        │   normals → micro-relief│                       │
        └───────────┬─────────────┘                       │
                    │                                     │
                    ▼                                     ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ 6. Composite                                                 │
        │    H = mask · ( w_bulk · depth + w_micro · relief            │
        │                 + w_photo · highpass(luma) )                 │
        │    + (1-mask) · plane                                        │
        │   weights driven by region map (5) when available            │
        └──────────────────────────┬───────────────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ 7. Tone curve / LUT (existing autofit + material profile)    │
        └──────────────────────────┬───────────────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ 8. Multi-pass planner (NEW)                                  │
        │    - Bulk pass    : low-res, deep ablation, defocused beam   │
        │    - Detail pass  : tight raster, mid power                  │
        │    - Polish pass  : single line-thickness, grayscale dither  │
        │    - Color passes : MOPA freq/power per material profile     │
        └──────────────────────────┬───────────────────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ 9. Export                                                    │
        │    - 16-bit PNG (one per pass)                               │
        │    - .lbrn project with all layers + parameters              │
        └──────────────────────────────────────────────────────────────┘
```

---

## 4. The opt-in pass stack (laser side)

Every export is a sequence of independently-toggleable passes. The planner generates this list from the heightmap + selected material profile + user toggles. Each entry becomes one layer in the `.lbrn` project with its own cut parameters.

| # | Pass | Default | Purpose | Source data | Toggle reason |
|---|------|---------|---------|-------------|---------------|
| 0 | **Pre-clean** | off | Light defocused pass to oxidise / pre-condition the surface (some stainless workflows) | const power | Skip if material is pre-prepared |
| 1 | **Form (bulk)** | on | Deep ablation of the largest height differences | low-res, high-power | Skip for shallow art-only engraving |
| 2 | **Cleanup** | on | Single fast pass with offset to remove burrs / re-cast slag from Form | derived: edge-dilation of Form mask | Skip on materials that don't burr (anodized Al) |
| 3 | **Detail** | on | Mid-power raster of full heightmap at material's native DPI | full-res 16-bit heightmap | Always on for relief work |
| 4 | **Shading** | on | Soft tonal pass: photo-luminance, dithered, very low power | photo luma → Riemersma dither | Skip for pure-relief outputs |
| 5 | **Polish** | optional | Final low-power tight raster to smooth interlace lines | full-res but defocus +0.5 mm | Skip on thick engravings |
| 6 | **Color 1..N** | optional | One pass per MOPA color band, parameters from material profile | per-color masks from luma quantisation or user paint | Skip entirely if monochrome; per-color toggles |
| 7 | **Signature** | optional | Tiny sub-mm relief mark in a corner (artist mark, machine ID) | const | Off by default, opt-in |

Key behaviours:
- **Each pass is its own `<CutSetting>`** in the LightBurn project; each pass is its own 16-bit PNG layer reference. The user can disable a pass in LightBurn after export without re-running our pipeline.
- **Time/cost estimate updates live** as passes are toggled.
- **Per-pass overrides** are stored in a `manifest.json` next to the .lbrn so the export is fully reproducible.

## 5. MOPA color profile ingestion

The user supplied five canonical reference projects:
```
LightBurn Colour Card/
  Colour20W-M7.lbrn2
  Colour30W-M7.lbrn2
  Colour60W-M7.lbrn2
  Colour80W-M7.lbrn2
  Colour100W-M7.lbrn2
```
Each is a LightBurn project containing ~100 `<CutSetting type="Scan">` entries — one per validated MOPA color — with the schema:
```xml
<CutSetting type="Scan">
  <index Value="N"/>
  <name Value="Cnn"/>
  <maxPower Value="50"/>      <!-- % -->
  <speed Value="500"/>        <!-- mm/s -->
  <frequency Value="300000"/> <!-- Hz -->
  <QPulseWidth Value="20"/>   <!-- ns -->
  <interval Value="0.0012"/>  <!-- mm (line spacing) -->
  <floodFill Value="1"/>      <!-- optional -->
</CutSetting>
```

### Phase 5a — Color profile importer (NEW, 2 h)
- New module `zoedepth/laser/profiles.py`:
  - `ColorEntry` dataclass mirroring the LightBurn schema fields above.
  - `MaterialProfile` = list of `ColorEntry` + tube wattage + machine name + source filename.
  - `load_lightburn_card(path: Path) -> MaterialProfile` — XML parser using `xml.etree.ElementTree`, indexed by `name` ("C00"…"Cnn") and by hashed parameter tuple for de-dup.
  - `load_all_profiles(dir: Path) -> dict[str, MaterialProfile]` — auto-loads every `*.lbrn2` file in a directory at startup (default: `LightBurn Colour Card/`).
  - `ColorEntry.thumbnail` — extracted from the project's `<Thumbnail>` base64 PNG when present, exposed in the UI as a swatch.
- New module `zoedepth/laser/color_picker.py`:
  - `match_color(rgb: tuple[int,int,int], profile: MaterialProfile) -> ColorEntry` — nearest-neighbour in CIELAB on the swatch thumbnails (when available) or by user mapping (fallback).
  - `quantise(image: np.ndarray, profile: MaterialProfile, k: int = 8) -> mask_per_color` — k-means in LAB with profile's available colors as forced centroids; returns one binary mask per color used.
- Export integration: each color in the quantisation produces one Color pass in the stack with cut parameters lifted *verbatim* from the imported entry. **No machine parameter is ever invented.**
- Tests:
  - `test_lightburn_loader.py` — round-trip parse all five supplied cards; assert ≥ 50 entries each, all numeric fields parse, no NaN.
  - `test_color_quantisation.py` — synthetic 4-color image → assert exactly 4 masks produced and reassemble within ε of original.

## 6. Live-preview architecture

The wizard and the studio are two presentations of the same in-memory pipeline graph.

```
Pipeline = ordered list of Stage objects
Stage    = (id, fn, inputs, outputs, params, cache_key)
```
A `PipelineRunner` evaluates only stages whose `(inputs ∪ params)` hash has changed; cached outputs of unchanged upstream stages are reused. All stages emit a 512² preview tensor in addition to their full-res output, so the UI can refresh on parameter changes in well under a second without re-running depth/normals.

Heavy stages (Marigold, ControlNet, Hunyuan3D) are explicitly **non-reactive**: changing their params dirties the cache but does NOT auto-recompute; a "Recompute heavy" button per stage gives the user control of GPU time.

New modules (Phase 5b, 4 h):
- `zoedepth/laser/pipeline.py` — `Stage`, `Pipeline`, `PipelineRunner`, hash-based cache.
- `zoedepth/laser/preview.py` — downscale-on-write helpers, ROI preview support (preview a 256² crop at full DPI for material-test workflows).
- UI: every accordion gets a thumbnail strip showing its stage's current 512² output; sliders bind through the runner.

## 7. Model selection

| Stage | Default | Params | License | Why |
|---|---|---|---|---|
| Subject mask | **BiRefNet (general 1024²)** | ~220 M | MIT | SOTA dichotomous segmentation; 17 FPS on 4090, 3.45 GB VRAM at fp16; ONNX-able. RMBG-2.0 is the same arch but non-commercial. |
| Mask fallback (CPU/low-VRAM) | rembg + `u2net_human_seg` | ~170 M | MIT | Pure-CPU, fast, "good enough" for testing. |
| Bulk depth | **Depth-Anything-V2-Base** | 97.5 M | Apache-2.0 | 10× faster than diffusion; clean affine-invariant depth; HF Transformers native. |
| Bulk depth (max quality) | Depth-Anything-V2-Large | 335 M | CC-BY-NC-4.0 | Used by ReliefGenerater; strictly better detail but non-commercial. |
| Normals | **DSINE** | 70 M | MIT | CVPR'24 oral; piecewise-smooth crisp normals; small enough for 4 GB GPU. |
| Normals (max quality) | Marigold-Normals v1-1 | ~900 M (SD2) | OpenRAIL++-M | Diffusion-based; slow but cleanest; used as opt-in. |
| Body-part seg (portraits) | **Sapiens-Seg-0.3B-lite** | 0.3 B | Sapiens License (research) | ECCV'24 best-paper candidate; 1024² native; 28 body parts → per-region depth weights. |
| Delighting (advanced) | Marigold-IID-Appearance | ~900 M | OpenRAIL++-M | Removes baked-in shadows so depth/normals see geometry only. |
| Super-res preprocessing | Real-ESRGAN x4plus | 17 M | BSD-3 | For inputs < 1024 px short side. |

---

## 8. Phased delivery

> Keep the studio working at every phase. Tests must stay green (currently 105/105).

### Phase 0 — Stop the bleeding (15 min, no new deps)
- `zoedepth/laser/autofit.py`: change suggestion `detail_mode="both", detail_strength=0.4` → `detail_mode="highpass", detail_strength=0.10`.
- `ui/mopa_studio.py`: dropdown labels — mark `luminance` and `both` as "(experimental — see plan)"; default `highpass`.
- Update `tests/test_autofit.py` expected values.
- Acceptance: existing skull-photo test render no longer exhibits ink-blot face.

### Phase 1 — Swap depth backbone (1–2 h)
- New module `zoedepth/laser/backbones/depth_anything.py` exposing `DepthAnythingV2Backbone` with `.predict(rgb_uint8) -> float32 depth (H,W)`.
- Refactor `service.py` to take a `DepthBackend` enum: `zoe-n` (current), `dav2-base` (new default), `dav2-large` (opt).
- Lazy-load weights from HF on first call; cache to `~/.cache/mopa/`.
- Tests: `test_backbone_dav2.py` (mock weights, check shape + dtype + monotonicity on a synthetic ramp).
- Risk: VRAM. Base = ~600 MB; Large = ~1.4 GB. Quadro P2000 4 GB easily handles Base.

### Phase 2 — Subject mask stage (1 h)
- `zoedepth/laser/subject_mask.py`: `SubjectMasker(backend="birefnet"|"rembg")` returning float32 alpha in [0,1].
- New step in `heightmap.py`: hard-clamp depth to plane outside the mask (`np.where(mask < 0.5, plane_value, depth)`); feather the boundary by N px (UI-configurable, default 3 px).
- UI: new accordion "Subject isolation" with backend dropdown + feather + manual override (upload custom alpha).
- Tests: `test_subject_mask.py` — synthetic foreground/background test image, assert mask separates them and final heightmap has flat background.

### Phase 3 — Normals + Frankot-Chellappa (3–4 h)
- `zoedepth/laser/normals.py`: `NormalEstimator(backend="dsine"|"marigold")` → `(N,3)` unit vectors per pixel.
- `zoedepth/laser/frankot_chellappa.py`: pure NumPy FFT integration of `(p, q) = (-Nx/Nz, -Ny/Nz)` → height. ~50 ms for 1024². Symmetric padding to suppress edge ringing. Self-contained, fully unit-testable on synthetic surfaces (sphere, ramp, sinusoid).
- Compose with bulk depth: `H = mask * (w_bulk * dav2_depth + w_micro * fc_relief)` with `w_bulk + w_micro = 1`.
- Default weights: 0.7 / 0.3, exposed in UI as a single "Detail vs. Form" slider.
- Tests:
  - `test_frankot_chellappa.py`: synthetic hemisphere normals → integrate → recovered surface within ε of analytic sphere.
  - `test_normals_pipeline.py`: end-to-end with a fixture image.

### Phase 4 — Per-region weighting via Sapiens (optional, 2–3 h, opt-in)
- `zoedepth/laser/regions.py`: `RegionSegmenter` returning a label map (face / hair / clothing / skin / accessory / bg).
- Region → weight table (face: high micro, low photo; hair: high photo, low micro; cloth: balanced; bg: zero).
- Gated behind `--enable-portrait-mode` flag and a UI checkbox; only activates when Sapiens detects ≥ 1 person.

### Phase 5 — Pass-stack planner, color-profile importer, live preview, .lbrn writer (6–8 h)
- **5a (above)** — Color profile importer for the supplied LightBurn cards.
- **5b (above)** — `pipeline.py` + `preview.py` reactive runner with cached stages and 512² previews.
- **5c — Pass planner** in `zoedepth/laser/stages.py`:
  - `EngravingPass(id, name, png_path, cut_setting: ColorEntry, enabled: bool, depends_on: list[str])`.
  - `plan_passes(heightmap, photo, profile, user_toggles) -> list[EngravingPass]` — produces the ordered stack from §4, honours the user's enable/disable map, derives Cleanup mask from Form's edge dilation, derives Color masks via `color_picker.quantise`.
  - Each pass carries its own preview; the UI renders the stack as a vertical strip of toggle-able cards with thumbnails, est. burn time, and "jump to settings" buttons.
- **5d — `lbrn_writer.py`** — emit a LightBurn `.lbrn2` (XML) project that:
  - References each pass's PNG as an image layer.
  - Embeds the verbatim `<CutSetting>` for each color pass (lifted from the imported card).
  - Writes a `<Thumbnail>` of the assembled preview.
  - Sets `MaterialHeight`, `MirrorY`, etc. from the source card.
- Tests:
  - `test_pass_planner.py` — toggle every pass on/off and assert the stack length / dependency integrity.
  - `test_lbrn_writer.py` — round-trip: write project, parse it back, assert every CutSetting matches the source profile bit-exactly.
  - `test_pipeline_caching.py` — change a downstream param, assert upstream stages are not re-run.

### Phase 6 — Wizard UI + reactive Studio (4–6 h)
- New tab "Wizard" with five pages, each backed by the live-preview runner:
  1. **Upload** — drop image, auto-orient, preview crop with face-aware framing.
  2. **Subject** — show mask overlay; brush-touch-up; soft/hard edge slider.
  3. **Form & Detail** — single "Detail vs. Form" slider (drives w_bulk/w_micro), with side-by-side preview of pure-depth vs. depth+normals.
  4. **Material & Passes** — material card picker → live pass-stack with toggles for every entry in §4 (Pre-clean, Form, Cleanup, Detail, Shading, Polish, Colors…, Signature). Each toggle live-updates the time estimate.
  5. **Review & Export** — assembled preview, manifest summary, file list, single Export button.
- The Studio (current page) is renamed "Advanced" and is **the same pipeline** — just every stage's accordion is open and every param is exposed.

---

## 9. Additional improvements (catalog of next-rung wins)

Ranked by **impact / effort**. ⭐ = recommended for first follow-up after Phase 5.

### Image preprocessing
- ⭐ **Real-ESRGAN x4plus super-resolution** for sub-1024 px inputs. Most engraving-quality losses come from low-res photos; SR before depth/normals lifts the entire pipeline. (BSD-3, 17 M params, < 0.5 s/image on GPU.)
- ⭐ **CLAHE + auto white-balance** as zero-cost preprocessing — DAv2 is more stable on well-exposed images.
- **CodeFormer / GFPGAN face restoration** for low-quality faces (compressed selfies, old photos). Faces are the highest-stakes area; even subtle restoration prevents pore-noise from dominating the relief.
- **Marigold-IID-Appearance "delighting"** to extract albedo only, removing baked-in shadows from the photo before normal estimation. Critical for outdoor / harsh-flash photos.
- **Auto-orient via face detection** — find the largest face, rotate so eyes are level, crop to medallion aspect.

### Depth / normals quality
- **Multi-resolution depth fusion** (PromptDA technique): run DAv2 at 512², 1024², 2048² and combine — global plane from low-res, edges from high-res. ~3× compute cost but visibly crisper silhouettes.
- **Bilateral cross-filtering**: use the photo as a guide image for an edge-preserving filter on the depth map. Keeps depth edges aligned with photo edges.
- **Stable-Normal v0-1** (Apache-2 alternative to Marigold-Normals) for users who want a permissive license but better-than-DSINE quality.
- **Test-time augmentation ensemble**: 4 rotations / horizontal flips → median per pixel. Adds 4× compute but removes single-pass artifacts.

### Segmentation / interaction
- **SAM-2 click-to-mask** for power users: click on subject → instant mask, click on accessories → secondary masks with their own depth weights.
- **Sapiens-Pose** keypoints to auto-place vignette / inset borders for medallion templates.
- **Face-parsing (BiSeNet-FP)** for sub-region weighting on faces (eyes flatter than nose, lips slightly raised, etc.) without the full Sapiens download.

### Generative augmentation (use with care; can hallucinate)
- **ControlNet-Depth + bas-relief LoRA**: feed the photo + depth, get back a stylised relief render; depth-estimate that → final geometry. This is the closest path to actually matching the "skull medallion" reference because it learns the relief style transformation.
- **Hunyuan3D-2-mini** (0.6 B, image → 3D mesh): generate a textured mesh, render orthographically with a Z-buffer → heightmap. The "luxury" path. Heavy (6 GB VRAM minimum) but unbeatable for jewelry-class quality on simple subjects.
- **Reference-style transfer**: user uploads a target relief style ("make it look like THIS skull"), we learn a small image-to-image transformation. Implementation: IP-Adapter on a stylised relief base.

### Heightmap post-processing
- **Riemersma dithering** instead of Floyd-Steinberg for the polish pass. More uniform laser-pulse density, fewer worm-like artefacts on smooth gradients.
- **Anisotropic median + bilateral chain** to remove sensor noise without blurring relief edges.
- **Histogram matching to a reference engraving**: if the user has a "good engraving" photo, match the depth histogram of new outputs to it.

### MOPA / laser physics
- **Anneal-color physics LUT**: model the titanium-oxide layer thickness vs. (frequency, power, speed, defocus, fluence) for stainless steel. Each color stage is a target thickness; the planner picks the parameters. Requires per-machine calibration ramp.
- **Calibration auto-fit**: burn the existing ramp once, photograph it on a flat-bed scanner, OpenCV extracts the patches, fits a polynomial to (depth_command → measured_optical_density). Fully closed-loop calibration in one button.
- **Per-pass time estimator**: integrate over the heightmap to compute total burn time per pass; warn if > N hours; suggest dpi/speed reductions.
- **Cooling-aware pass ordering**: long deep passes first, then color passes after a cool-down beep.

### Project / UX
- **Material profile sharing**: profiles are JSON with a small reference image; an `import` / `export` button + a community gallery.
- **Reproducibility manifest**: every export ships a `manifest.json` capturing model versions, hashes, settings, photo EXIF, profile id, git commit. Re-runs are bit-exact.
- **Undo stack on settings**: trivial in Gradio with state callbacks; massively reduces user frustration during iteration.
- **CLI parity**: every studio knob also a flag on `python -m mopa.cli`. Enables batch processing.
- **Test harness with golden images**: 10 reference photos × 4 material profiles → deterministic PNG hashes stored in repo. Catches regressions instantly.

### Performance
- **ONNX / TensorRT export** for BiRefNet + DSINE + DAv2 — already supported by upstream; potential 3-5× speedup on the P2000.
- **fp16 everywhere** (already supported by all chosen models). Halves VRAM, ~zero quality loss.
- **Batched processing of multiple photos** for shop owners who do production runs.

### Safety / correctness
- **EXIF stripping by default** on exports (privacy when sharing engraving files).
- **NSFW classifier** on uploads if shipped publicly — engraving services are often public-facing.
- **Watermark / signature mode**: optional sub-mm relief signature in a corner.

---

## 9b. Phase 9 — Angular + SignalTree + PrimeNG SPA migration

> Replaces the Gradio-based [`ui/mopa_studio.py`](ui/mopa_studio.py) and [`ui/mopa_wizard.py`](ui/mopa_wizard.py) front-ends with a real desktop-class single-page app. The Python pipeline (everything under `zoedepth/laser/`) becomes a headless service exposed over HTTP + WebSocket; the Gradio UIs stay shipped during the transition as the fallback surface.

### Why migrate
- **Layout limits.** Gradio can't do drag-resizable splitter panes, true overlay split-view, or a persistent thumbnail strip without hand-rolled HTML hacks.
- **State explosion.** ~30 calibration knobs feeding one preview pipeline is past the point where a flat handler signature scales — every "add a control" turn already requires touching the handler tuple, the `inputs=[…]` list, and the override-keys constant in three places.
- **Reactive granularity.** Today every slider drag re-evaluates the full handler. We want leaf-level reactivity so only the affected pipeline substage recomputes.
- **Polish ceiling.** Toasts, modal confirms, drag-and-drop file uploads with chunking, image-compare scrubbers — all stock components in PrimeNG, all bespoke in Gradio.

### Stack

| Layer | Pick |
|---|---|
| Framework | Angular **latest stable** (track current release; standalone components + signals throughout) |
| Components | **PrimeNG** (Aura theme) + PrimeIcons + PrimeFlex |
| State | **SignalTree** — single root tree, services expose typed sub-trees |
| HTTP | `HttpClient` + a thin `ApiService` wrapper |
| Realtime | `rxjs/webSocket` for inference progress + preview push |
| Forms | Reactive Forms bound to tree leaves via two-way signal helpers |
| Build / Test | Angular CLI (esbuild) + Jest + Playwright (E2E) |
| Backend | **FastAPI** wrapping the existing `zoedepth.laser.service` |

PrimeNG components that map directly to current pain points:
- **Splitter** — drag-resizable left/right panels (Gradio can't).
- **ImageCompare** — built-in before/after slider with draggable divider (replaces hand-rolled split-view).
- **Galleria** — thumbnail strip for export history.
- **Slider / Knob / ColorPicker / Dropdown / Accordion / TabView** — every calibration control.
- **FileUpload** with chunking + progress.
- **Toast / ConfirmDialog / ProgressBar** — polish.

### Why SignalTree (not plain signals, not NgRx)
- Studio state is naturally tree-shaped: `profile.tone.lut`, `profile.smoothing.bilateral`, `settings.export.naming`, `session.preview.shaded`, `session.history[]`. SignalTree maps 1:1 onto that without per-slice store boilerplate.
- Granular reactivity scoped to the leaf that changed — fewer wasted preview recomputes.
- Snapshot-to-JSON for "save profile" / "restore session" is a one-liner; same for hydration.
- Maintained in-house — bug-fix turnaround = same day.
- Undo/redo path stays open via tree-snapshot ring buffer.

### Suggested root tree shape

```ts
const studioTree = signalTree({
  session: {
    image: null as File | null,
    preview: { shaded: null, eightBit: null, posterized: null } as Preview,
    busy: false,
    progress: 0,
  },
  profile: { /* mirrors profiles.py schema */ },
  settings: {
    input: { clahe: 8, specularThreshold: 240, maxDim: 1024 },
    tone: { lut: [...], targetDepthUm: 0 },
    smoothing: { /* ... */ },
    edge: { /* ... */ },
    dither: { levels: 0 },
    posterize: { passes: 0 },
    phase8: {
      preUpscale: false, resolver: 'lanczos', targetPx: 2048,
      multires: false, reliefStrength: 0,
      writeStl: false, stlHeightScale: 0.1, stlBaseThickness: 0.05, stlSubsample: 4,
    },
    export: { dir: '', naming: 'overwrite', writePreview: true, writeRamp: false },
  },
  history: [] as ExportRecord[],
});
```

### Phased delivery

#### Phase 9a — Headless service surface (1–2 days)
- New `mopa_service/` Python package (sibling of `zoedepth/`) with FastAPI app:
  - `POST /preview` — multipart image + settings JSON → preview PNG bytes + heightmap stats.
  - `POST /export` — same payload → ExportBundle JSON (paths + manifest).
  - `POST /autofit` — image + profile → suggested overrides.
  - `GET /profiles` / `GET /resolvers` / `GET /backends` — enumerate registries.
  - `WS /progress/{job_id}` — push stage-level progress events.
- Reuse `zoedepth.laser.service.InferenceService` verbatim — the FastAPI handlers are thin wrappers over the same `_service()` singleton the Gradio UIs already call.
- Tests: `tests/test_api_*.py` using FastAPI's `TestClient`. Reuse the `_StubService` pattern from `test_studio_phase8.py`.

#### Phase 9b — Angular workspace bootstrap (½ day)
- `frontend/` directory at repo root, `ng new` against the **latest stable Angular CLI** (`npm i -g @angular/cli@latest` then `ng new`); standalone components + signals + Jest.
- Pin the major version in `package.json` and add a CI guard that fails if `@angular/core` falls more than one minor behind upstream latest.
- Add PrimeNG (latest), PrimeIcons, PrimeFlex, SignalTree (local dep or git submodule).
- CI: add `npm test` and `npm run build` to the existing GH Actions matrix.

#### Phase 9c — Studio parity (3–4 days)
- Implement panes in this order, each one fully replacing its Gradio counterpart before moving on:
  1. **Upload + Preview** — drop-zone (FileUpload), Splitter, ImageCompare for original ↔ shaded preview.
  2. **Calibration sidebar** — every slider/dropdown bound to its SignalTree leaf; debounced auto-preview.
  3. **Profile picker + Material card** — Dropdown + Accordion of color-pass cards.
  4. **Pass stack viewer** — Galleria-style strip showing each `EngravingPass` as a card with its thumbnail + cut-setting summary.
  5. **Export panel** — destination picker, naming-policy dropdown, write-flags, .stl options, .lbrn2 toggle.
- Each pane gets a Jest spec for state binding + a Playwright E2E for the user flow.

#### Phase 9d — Wizard parity (2 days)
- Five PrimeNG `Stepper` pages mirroring the Gradio wizard: Upload → Subject → Form & Detail → Material & Passes → Review & Export.
- Auto-advance on success, allow free navigation; persist wizard state in the same SignalTree under `session.wizard`.

#### Phase 9e — Phase 8 parity in the SPA (1 day)
- Pre-upscale toggle on the Upload pane.
- SAM-2 click-to-mask with PrimeNG `Image` + click events on the Subject pane (the Gradio Studio intentionally skipped this — the SPA recovers it).
- Multi-resolution fusion + bas-relief slider on the Detail pane.
- ControlNet / Hunyuan3D opt-in cards (gated by `allow_nc_weights`).
- .stl options on the Export pane.

#### Phase 9f — Cutover & deprecation (½ day)
- Both UIs ship side-by-side for one release. Default startup script launches the SPA + FastAPI; `--legacy-gradio` flag launches the existing Blocks app.
- Delete the Gradio UIs the release after, keeping `ui/gradio_*.py` *demos* (the original ZoeDepth gradio examples) untouched.

### Tests required to call Phase 9 done
- `tests/test_api_preview.py` / `test_api_export.py` / `test_api_autofit.py` — full HTTP round-trips through the FastAPI surface using stub services.
- `frontend/src/**/*.spec.ts` — Jest specs for SignalTree wiring of every Studio pane.
- `frontend/e2e/*.spec.ts` — Playwright flows for upload → preview → export and wizard happy-path.
- Existing 326 Python tests stay green throughout.

### Risks & mitigations
- **Two UIs to maintain during cutover.** Mitigation: SPA only calls the public service surface; Gradio keeps calling internal helpers. They never share UI code, only the underlying `InferenceService`.
- **Local file paths vs. browser sandbox.** The SPA can't write to arbitrary local dirs. Mitigation: FastAPI does all disk writes; the SPA streams previews and shows file links the user can open externally (or downloads via `Content-Disposition`).
- **Real-time preview latency over HTTP.** Mitigation: the `/preview` route already returns in < 200 ms for cached substages; SPA debounces to 150 ms; full-quality renders go through `/export`.
- **SignalTree is in-house.** Mitigation: it's our code — bugs are same-day fixes; no third-party rate-limit on PRs.

### Out of scope for Phase 9 (deferred)
- Multi-user / cloud deploy (still strictly local).
- Auth / per-user profiles.
- Live collaboration on a single project.
- Tauri / Electron packaging — browser-launched is fine for v1; revisit if "double-click to launch" becomes a usability blocker.

---

## 10. Out of scope (intentionally)

- Vector / line-art conversion (engravers already have tooling).
- Video → engraving (single-frame only).
- Cloud / multi-user — strictly a local desktop tool.
- Training new models — only inference on published checkpoints.

---

## 11. Decisions & open questions

### Decided
- **Default machine**: **60 W MOPA (M7)**. `LightBurn Colour Card/Colour60W-M7.lbrn2` is the autoload profile; the other four cards are selectable from a dropdown.
- **Color masking**: **Both strategies**. Automatic LAB k-means against the profile runs by default (zero-effort path); a brush/click-to-paint overlay lets the user override or assign specific colors to specific regions. The two share the same `mask_per_color` data structure so the pass-planner downstream is unchanged either way.
- **License policy**: ship **Apache-2 / MIT defaults** (DAv2-Base, DSINE, BiRefNet, rembg, Real-ESRGAN). The CC-BY-NC and RAIL-licensed models (DAv2-Large, RMBG-2.0, Sapiens, Marigold-*, Hunyuan3D-2) are **opt-in** behind a one-time "Enable non-commercial models" toggle in Settings, with the license summary shown inline. Default install never silently downloads NC weights.
- **Hardware floor**: **warn, never block.** On startup we probe `torch.cuda.mem_get_info()` and tag each backend with a min-VRAM hint (e.g. Marigold ≥ 6 GB, Hunyuan3D ≥ 8 GB). If the user picks a backend over their VRAM headroom they get an inline yellow banner ("This will likely OOM on your 4 GB GPU — proceed?") with a Proceed button; we never disable a control.
- **Output target**: heightmap PNG always written. **`.lbrn2` project export is a checkbox** in the Export panel ("Emit LightBurn project (.lbrn2)"), default ON. When unchecked, only the per-pass PNGs + `manifest.json` are written.

### Still open
*(none — proceeding to Phase 0)*
