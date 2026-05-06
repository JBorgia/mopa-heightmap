# MOPA Heightmap Studio — Planning Document

A focused desktop app that turns a photo into a LightBurn 3D Sliced–ready heightmap optimized for **JPT MOPA fiber galvo** engraving on metal.

Built as a thin UI + export layer on top of the existing ZoeDepth model and the new `mopa` pipeline. **The model is not modified.**

---

## 1. Goals & Non-Goals

### Goals (v1)
- One-shot pipeline: drop an image → preview → download LightBurn-ready PNG.
- Run fully **local** on the operator's workstation. No cloud, no auth, no upload limits.
- Reproducible exports: every PNG ships with a sidecar settings JSON.
- Material-aware: ship MOPA presets for brass, stainless, aluminum, copper.
- Provide a calibration workflow (ramp PNG) so operators can map gray → real depth on their machine.
- Honest UX: no "AI magic" claims. The app exposes percentile clipping, gamma, smoothing, polarity, and background flattening as direct controls.

### Non-Goals (v1)
- No model retraining or architecture changes.
- No multi-user / hosted / authenticated deployment.
- No `.lbrn2` project file generation (deferred to v2).
- No automatic subject segmentation (deferred to v2).
- No CAM-level toolpath generation. The app produces an image; LightBurn does the slicing.
- No mobile UI.

### Success Criteria
- An operator can open the app, drop a coin photo, pick `mopa_60w_brass`, click **Export**, and load the resulting PNG into LightBurn 3D Sliced without further edits.
- Re-running the same image with the same profile produces a byte-identical PNG.
- Calibration ramp engraved on the operator's actual machine produces visible, distinguishable steps from 0–255.

---

## 2. Target User & Workflow

**User:** A single operator running a JPT MOPA fiber galvo (typically 30W–100W) with LightBurn, engraving relief on small metal parts (coins, medallions, plaques, knife scales, watch backs).

**Current pain points the app solves:**
1. ZoeDepth output is metric camera depth, not engraving relief — needs remapping.
2. LightBurn 3D Sliced expects `darker = more passes = deeper`, opposite of intuition.
3. Background depth noise wastes engraving time on coins/medallions.
4. Each material + lens + power combo needs different gray → depth response.

**End-to-end workflow:**
```
photo.jpg
  → MOPA Heightmap Studio (this app)
  → coin_lightburn.png (8-bit grayscale)
  → LightBurn → Image → Mode: 3D Sliced
  → place on jig, run job
```

---

## 3. Architecture

### 3.1 Layered design

```
┌─────────────────────────────────────────────┐
│  UI Layer        ui/mopa_studio.py          │  ← Gradio Blocks
│                  ui/app.py (entrypoint)     │
├─────────────────────────────────────────────┤
│  Service Layer   mopa/service.py  │  ← orchestrates infer + process + export
├─────────────────────────────────────────────┤
│  Pipeline Layer  mopa/            │
│                  ├─ heightmap.py            │  (already built)
│                  ├─ preview.py              │  (already built)
│                  ├─ profiles.py             │  (already built)
│                  ├─ tiling.py               │  (already built)
│                  ├─ masks.py                │  (already built)
│                  └─ exporter.py             │  ← new: bundles outputs
├─────────────────────────────────────────────┤
│  Model Layer     hubconf.py + zoedepth/     │  ← UNCHANGED
└─────────────────────────────────────────────┘
```

The CLI (`apps/mopa2lightburn.py`) and the UI both call into the **service layer**. Same code path, two front-ends.

### 3.2 Process model

- Single Python process. Gradio runs an internal web server bound to `127.0.0.1`.
- Model is loaded **once** at startup, kept resident in GPU memory.
- Each export call is synchronous; Gradio's queue handles serialization.
- No background workers, no Celery, no Redis. Not needed for one operator.

### 3.3 Data flow

```
PIL.Image (RGB, uint8)
  → model.infer_pil()                         → np.float32 depth (H, W)
  → normalize_depth(near%, far%)              → [0, 1] float32
  → orient_for_lightburn(black_is_deep)       → [0, 1] float32
  → apply_tone_curve(gamma, contrast, ...)    → [0, 1] float32
  → flatten_background(threshold) [opt]       → [0, 1] float32
  → smooth_heightmap(bilateral|gaussian)      → [0, 1] float32
  → sharpen_heightmap(unsharp_mask)           → [0, 1] float32
  → exporter.bundle()
       ├─ <stem>_lightburn.png   (uint8 L)    ← LightBurn input
       ├─ <stem>_master16.png    (uint16 I;16)
       ├─ <stem>_preview.png     (RGB)
       ├─ <stem>_ramp.png        (uint8 L)    [optional]
       └─ <stem>_settings.json
```

---

## 4. UI Specification

### 4.1 Layout (single-page Gradio Blocks)

```
┌──────────────────────────────────────────────────────────────────┐
│  MOPA Heightmap Studio                       [Calibration ramp] │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────────────────────────────────┐   │
│  │              │  │  Profile:    [mopa_60w_brass    ▼]     │   │
│  │   Input      │  │  Model:      [ZoeD_NK ▼]               │   │
│  │   Image      │  │  ──────────── Depth ────────────────   │   │
│  │  (drop here) │  │  Near %  [── 5 ──]   Far %  [── 95 ──] │   │
│  │              │  │  ──────────── Tone  ────────────────   │   │
│  └──────────────┘  │  Gamma   [0.72]  Contrast [1.00]       │   │
│                    │  Deep limit [0.04]  Surface [0.96]     │   │
│  ┌──────────────┐  │  ──────────── Cleanup ──────────────   │   │
│  │              │  │  Smooth  [bilateral ▼]  Strength [.08] │   │
│  │   Shaded     │  │  Sharpen [0.20]                        │   │
│  │   Preview    │  │  ──────────── Background ───────────   │   │
│  │              │  │  [x] Flatten background                 │   │
│  └──────────────┘  │  Threshold [0.88]  Value [1.00]        │   │
│                    │  ──────────── Polarity ─────────────   │   │
│  ┌──────────────┐  │  (•) Black is deep  ( ) White is deep  │   │
│  │  Histogram   │  │  ────────────────────────────────────  │   │
│  └──────────────┘  │       [ Preview ]   [ Export ]         │   │
│                    └────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│  Downloads:  [lightburn.png] [master16.png] [settings.json]     │
│  Status:     Ready · Last export 0.84s · Device: cuda:0         │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Components & wiring

| Control | Type | Bound to | Default |
|---|---|---|---|
| Input image | `gr.Image(type="pil")` | service input | — |
| Profile | `gr.Dropdown` | loads YAML, repopulates sliders | `mopa_60w_brass` |
| Model | `gr.Dropdown` | model loader (deferred swap) | `ZoeD_NK` |
| Near % | `gr.Slider(0, 25, step=0.5)` | `near_percentile` | 5 |
| Far % | `gr.Slider(75, 100, step=0.5)` | `far_percentile` | 95 |
| Gamma | `gr.Slider(0.3, 2.0, step=0.01)` | `gamma` | 0.72 |
| Contrast | `gr.Slider(0.5, 2.0, step=0.01)` | `contrast` | 1.0 |
| Deep limit | `gr.Slider(0, 0.3, step=0.01)` | `deep_limit` | 0.04 |
| Surface limit | `gr.Slider(0.7, 1.0, step=0.01)` | `surface_limit` | 0.96 |
| Smooth | `gr.Dropdown([none, bilateral, gaussian])` | `smooth` | bilateral |
| Smooth strength | `gr.Slider(0, 0.3, step=0.005)` | `smooth_strength` | 0.08 |
| Sharpen | `gr.Slider(0, 1.0, step=0.01)` | `sharpen` | 0.2 |
| Flatten BG | `gr.Checkbox` | `flatten_background` | True |
| BG threshold | `gr.Slider(0.5, 1.0, step=0.01)` | `background_threshold` | 0.88 |
| Polarity | `gr.Radio([Black is deep, White is deep])` | `black_is_deep` | Black |
| Preview button | `gr.Button` | runs pipeline, updates 3 image panels |
| Export button | `gr.Button` | runs pipeline, writes to `outputs/`, returns file links |
| Calibration ramp | `gr.Button` | writes `outputs/calibration_ramp.png` |

### 4.3 UX behavior rules

- **Preview vs. Export**: Preview runs the full pipeline but writes nothing to disk. Export writes the bundle and exposes downloads.
- **Profile change**: replaces all slider values, but does **not** auto-rerun. Operator clicks Preview.
- **Model change**: shows a "loading…" spinner, swaps the resident model, then re-enables controls.
- **Cached depth**: if input image hash unchanged since last preview, skip inference and just re-run post-processing. This makes slider tweaks feel instant.
- **Errors**: shown in the status bar, never as popup modals.

---

## 5. Service Layer Specification

New file: `mopa/service.py`

```python
class HeightmapService:
    def __init__(self, model_name: str = "ZoeD_NK", device: str | None = None): ...
    def switch_model(self, model_name: str) -> None: ...
    def list_profiles(self) -> list[str]: ...
    def load_profile(self, name: str) -> dict: ...
    def preview(self, image: PIL.Image.Image, settings: dict) -> PreviewResult: ...
    def export(self, image: PIL.Image.Image, settings: dict, output_dir: Path,
               stem: str, include_ramp: bool = False) -> ExportBundle: ...
    def make_calibration_ramp(self, output_path: Path) -> Path: ...

@dataclass
class PreviewResult:
    grayscale: PIL.Image.Image       # uint8 L → RGB for display
    shaded:    PIL.Image.Image       # RGB shaded relief
    histogram: PIL.Image.Image       # RGB histogram strip
    elapsed_s: float
    device:    str

@dataclass
class ExportBundle:
    lightburn_png: Path
    master16_png:  Path
    preview_png:   Path
    settings_json: Path
    ramp_png:      Path | None
    elapsed_s:     float
```

**Caching contract:** the service keeps `(input_hash, depth_array)` for the most recent preview. `preview()` and `export()` both reuse this cache when the input image hash matches.

---

## 6. Exporter Specification

New file: `mopa/exporter.py`

Responsibilities:
- Compute consistent file stems (strip trailing `_lightburn`, etc., to avoid `_lightburn_lightburn.png`).
- Write 8-bit PNG with PIL `mode="L"`, no alpha, no ICC profile.
- Write 16-bit PNG with PIL `mode="I;16"`. Verified by reading back and asserting `dtype == uint16`.
- Write `settings.json` with: input path, image hash (sha256, first 16 hex), model name, device, profile name + raw profile dict, effective settings dict, app version, timestamp (UTC ISO-8601).
- Atomic writes: write to `*.tmp` then `os.replace`. Avoids LightBurn reading half-written files if it's polling a watch folder.

---

## 7. CLI Parity

`apps/mopa2lightburn.py` keeps working unchanged. After the refactor, both UI and CLI call `HeightmapService`. Anything the UI can do, the CLI can do via flags.

**New CLI subcommand to add:** `--gui` to launch the UI from the same script.

```bash
python apps/mopa2lightburn.py --gui
python apps/mopa2lightburn.py input.jpg --profile mopa_60w_brass --output out/coin.png
python apps/mopa2lightburn.py --make-ramp outputs/ramp.png
```

---

## 8. Profiles

Existing YAML files under `profiles/` stay the schema of record. Schema:

```yaml
name: <string, matches filename without extension>
machine: <string, human-readable>
lightburn_mode: "3D Sliced"
black_is_deep: <bool>

heightmap:                # passed to process_depth_to_heightmap()
  near_percentile: <0-50>
  far_percentile:  <50-100>
  gamma:           <0.3-2.0>
  contrast:        <0.5-2.0>
  midtone_boost:   <0-0.3>
  deep_limit:      <0-0.3>
  surface_limit:   <0.7-1.0>
  smooth:          <"none"|"bilateral"|"gaussian">
  smooth_diameter: <int>
  smooth_strength: <0-0.3>
  sharpen:         <0-1.0>
  sharpen_sigma:   <float>
  flatten_background:    <bool>
  background_threshold:  <0.5-1.0>
  background_value:      <0-1>

lightburn_starting_point: # informational only, not applied to image
  lens, speed_mm_s, power_percent, frequency_khz, pulse_width_ns,
  line_interval_mm, passes, angle_increment, cleanup_every_passes
```

**Validation:** `profiles.load_profile()` will gain a schema check that rejects unknown keys and out-of-range values, with a clear error message in the UI status bar.

---

## 9. Calibration Workflow

1. Operator clicks **Calibration ramp** → app writes `outputs/calibration_ramp.png` (11 steps: 0, 26, 51, … 255).
2. Operator runs the ramp in LightBurn 3D Sliced with the **starting-point settings from the chosen profile**.
3. Operator measures actual depth at each step with calipers / depth gauge.
4. (v2) Enters measurements back into the app; app fits a correction LUT and stores it in the profile YAML under `calibration_lut`.
5. Future exports apply the LUT after `apply_tone_curve` and before quantization.

v1 ships steps 1–3. The LUT input UI is v2.

---

## 10. File & Folder Layout (final state for v1)

```
mopa-heightmap/
├─ apps/
│  └─ mopa2lightburn.py            # CLI + --gui entrypoint
├─ ui/
│  ├─ app.py                      # rewritten: launches mopa_studio
│  └─ mopa_studio.py              # NEW: Gradio Blocks UI
├─ zoedepth/
│  ├─ laser/
│  │  ├─ __init__.py
│  │  ├─ heightmap.py             # exists
│  │  ├─ preview.py               # exists
│  │  ├─ profiles.py              # exists, gains validation
│  │  ├─ tiling.py                # exists
│  │  ├─ masks.py                 # exists
│  │  ├─ service.py               # NEW
│  │  └─ exporter.py              # NEW
│  └─ ... (model code, untouched)
├─ profiles/
│  ├─ mopa_60w_brass.yaml
│  ├─ mopa_60w_stainless.yaml
│  ├─ mopa_60w_aluminum.yaml
│  └─ mopa_60w_copper.yaml
├─ outputs/                       # gitignored; generated bundles live here
├─ tests/                         # NEW
│  ├─ test_heightmap.py
│  ├─ test_profiles.py
│  ├─ test_exporter.py
│  └─ fixtures/
│     └─ tiny_depth.npy
├─ environment.yml
└─ README.md
```

---

## 11. Dependencies

Already present: `torch`, `torchvision`, `numpy`, `opencv`, `PIL`, `pyyaml`.

**Add** (pip section of `environment.yml`):
- `gradio>=4.0,<5` — UI
- `pytest>=8.0` — tests

No Node, no Docker, no FastAPI for v1.

---

## 11a. Application Settings

User-level app settings live in `~/.mopa-heightmap/settings.json` (created on first launch). These are distinct from per-export profile settings — they're the operator's preferences for how the app itself behaves.

### 11a.1 Schema

```json
{
  "version": 1,
  "output": {
    "directory": "outputs",
    "naming": "overwrite",                  // "overwrite" | "timestamp" | "counter"
    "timestamp_format": "%Y%m%d_%H%M%S",     // used when naming = "timestamp"
    "keep_history": false                    // when true, never overwrite even in "overwrite" mode (auto-bumps to counter)
  },
  "preview": {
    "resolution_cap": 1024,                  // 0 = no cap, full resolution every preview
    "flip_aug": false,                       // skip flip augmentation in preview for speed
    "auto_rerun_on_slider_change": false     // v2: live preview while dragging
  },
  "inference": {
    "default_model": "ZoeD_NK",
    "device": "auto",                        // "auto" | "cuda" | "cuda:0" | "cpu"
    "flip_aug": true                         // applies to Export, not preview
  },
  "ui": {
    "open_browser_on_launch": true,
    "server_port": 7860,
    "theme": "default"
  }
}
```

### 11a.2 Output naming modes

| Mode | Behavior | Example |
|---|---|---|
| `overwrite` | Replace existing files of the same stem. | `coin_lightburn.png` |
| `timestamp` | Append a UTC timestamp before the suffix. | `coin_20260501_143022_lightburn.png` |
| `counter` | Append `_v2`, `_v3`, … incrementing past the highest existing. | `coin_v3_lightburn.png` |

`keep_history = true` forces non-destructive behavior even when `naming = "overwrite"` (treated as `counter`). This is the safety belt for accidental re-exports.

### 11a.3 Preview resolution cap

- `resolution_cap = 1024` (default): if the input's long edge exceeds 1024 px, the preview pipeline runs on a downscaled copy. **Export always uses full resolution regardless.**
- `resolution_cap = 0`: preview uses full resolution. Slower per slider tweak, but pixel-identical to the export.
- Rationale: ZoeDepth resizes internally; bilateral and unsharp passes scale O(pixels). Capping preview keeps the slider loop snappy without affecting final output quality.

### 11a.4 UI exposure

A **Settings** tab (or modal) exposes all of the above with the same control patterns as profile sliders. Changes write to `settings.json` immediately and take effect on the next operation. No restart required except for `server_port` and `device`.

### 11a.5 Loading order

```
defaults  →  ~/.mopa-heightmap/settings.json  →  CLI flags (per-invocation override)
```

Profiles are independent of app settings; they only override the heightmap pipeline parameters.

---

---

## 12. Performance Targets

| Stage | Target on RTX 30-series, 1024² input |
|---|---|
| Model load (cold) | < 15 s |
| Inference (`infer_pil`, no flip aug) | < 1.5 s |
| Inference (with flip aug) | < 3 s |
| Post-processing (full pipeline) | < 100 ms |
| Preview round-trip (cached depth) | < 150 ms |
| Export round-trip (cached depth) | < 400 ms |

CPU-only fallback: targets relax to ~10× slower. Acceptable for one-shot exports, painful for slider-tweaking. UI should warn when running on CPU.

---

## 13. Testing Strategy

`tests/` (pytest):

- **`test_heightmap.py`** — unit tests on the pipeline using a synthetic 64×64 depth gradient:
  - `normalize_depth` produces `[0, 1]`, monotonic.
  - `orient_for_lightburn(black_is_deep=True)` inverts.
  - `apply_tone_curve(gamma=1, contrast=1)` is identity within tolerance.
  - `flatten_background` only modifies pixels above the threshold.
  - `to_uint8` / `to_uint16` round-trip without overflow.
- **`test_profiles.py`** — every YAML in `profiles/` loads, passes schema validation, and contains every key referenced by `DEFAULT_SETTINGS`.
- **`test_exporter.py`** — exporter writes all 4 files, settings JSON parses, `_master16.png` reads back as uint16.
- **No model tests in CI.** The model download is too large; model behavior is upstream's responsibility. A manual `sanity.py` already exists for that.

---

## 14. Build Plan (Order of Operations)

Each step is a self-contained PR-sized change.

1. **Service layer** — extract orchestration from CLI into `HeightmapService`. CLI becomes a thin caller. No behavior change.
2. **Exporter** — move bundle-writing into `exporter.py` with atomic writes and richer settings JSON.
3. **Profile validation** — schema check + clear error messages.
4. **Tests** — add `tests/` with the suite from §13. Get CI-greenable on a CPU-only machine (skip model tests).
5. **UI v1** — `ui/mopa_studio.py` implementing §4. Wire everything to the service.
6. **CLI `--gui`** — launch the UI from `apps/mopa2lightburn.py --gui`.
7. **README pass** — replace the upstream ZoeDepth quickstart at the top with a "MOPA Heightmap Studio" quickstart. Keep the old material lower as "Underlying model".
8. **Caching** — input-hash → depth cache for fast slider tweaks.
9. **Polish** — CPU warning, status bar, error handling, atomic writes verified.

Stop here for v1.

---

## 15. Feature Backlog (Value-Ranked, Effort-Agnostic)

Full ranked list across the whole roadmap (v1.5 → v3). Used to plan phases in §24.

| Rank | ID | Feature | Phase |
|---:|---|---|---|
| 1 | F17 | Per-material gray→depth LUT | v1.5 |
| 2 | F38 | LUT input UI (type measured depths) | v1.5 |
| 3 | F23 | Max-depth-budget in microns | v1.5 |
| 4 | F25 | `.lbrn2` project export | v2 |
| 5 | F5  | Subject auto-mask (rembg/SAM) | v1.5 |
| 6 | F14 | Edge-aware joint-bilateral refinement | v1.5 |
| 7 | F49 | Region-of-interest brush | v3 |
| 8 | F7  | DPI/mm stamping in PNG (pHYs chunk) | v1.5 |
| 9 | F20 | Posterization preview (N passes) | v1.5 |
| 10 | F4  | Perspective rectification (ellipse → circle) | v2 |
| 11 | F12 | MiDaS + ZoeDepth ensemble | v2 |
| 12 | F2  | Specular highlight removal | v2 |
| 13 | F36 | A/B compare panel | v2 |
| 14 | F15 | Frequency separation (form vs. detail) | v2 |
| 15 | F30 | Multi-lens calibration (LUT per lens) | v2 |
| 16 | F18 | Pulse-width-aware tone curves (LUT per Q-pulse preset) | v2 |
| 17 | F22 | Min-feature-size guard at lens spot size | v1.5 |
| 18 | F46 | Live debounced slider preview | v1.5 |
| 19 | F11 | Bilateral inference TTA | v2 |
| 20 | F6  | Center + crop to aspect | v1.5 |
| 21 | F21 | Dithered 16→8-bit quantization | v1.5 |
| 22 | F32 | Batch mode | v1.5 |
| 23 | F34 | Re-run from `_settings.json` | v1.5 |
| 24 | F26 | Workpiece templates (coin / dog tag / Zippo lid) | v2 |
| 25 | F1  | Auto white balance + exposure normalization | v1.5 |
| 26 | F16 | Feathered tile blending | v2 |
| 27 | F31 | Watch folder | v2 |
| 28 | F47 | Hover tooltips on every control | v1.5 |
| 29 | F48 | Histogram overlay on threshold sliders | v1.5 |
| 30 | F53 | 3D mesh preview | v2 |
| 31 | F19 | Layer-separation export (N quantized PNGs) | v2 |
| 32 | F51 | Reset to profile defaults | v1 add-on |
| 33 | F42 | Engraving-time estimate + long-job warning | v1.5 |
| 34 | F50 | Pin/lock individual sliders across profile changes | v2 |
| 35 | F33 | Recipe history JSONL | v1.5 |
| 36 | F8  | EXIF orientation handling | v1 add-on |
| 37 | F52 | False-color heightmap preview | v1.5 |
| 38 | F54 | Side-by-side input vs. output panel | v1.5 |
| 39 | F39 | Ramp variants (log-spaced, fine-deep, per-angle) | v2 |
| 40 | F28 | Cleanup-pass interleaver export | v1.5 (now part of multi-pass; see §21) |
| 41 | F56 | PyInstaller one-file build | v3 |
| 42 | F24 | Symmetric mirror / radial cleanup for round subjects | v2 |
| 43 | F60 | Crash-report bundle | v3 |
| 44 | F35 | Undo/redo on slider changes | v2 |
| 45 | F59 | Local model-cache management UI | v3 |
| 46 | F57 | Dockerfile (CUDA base) | v3 |
| 47 | F3  | Shadow lift | v1.5 (bundled with F1) |
| 48 | F45 | Reflective-surface mode warning | v2 |
| 49 | F9  | Multi-image fusion (averaging) | v3 |
| 50 | F41 | Per-batch calibration metadata | v3 |
| 51 | F55 | Drag-and-drop profile YAML install | v2 |
| 52 | F37 | Bookmark / favorite within profile | v3 |
| 53 | F44 | Material × wavelength reminder | v3 |
| 54 | F27 | Hatch-angle scheduler | Skip (LightBurn native) |
| 55 | F13 | Depth-from-focus assist | Skip |
| 56 | F10 | Focus stacking input | Skip |
| 57 | F40 | Auto-LUT from microscope photo | Skip |
| 58 | F58 | Auto-update channel | Skip |
| 59 | F43 | Power × pulse-width safety check | Skip (false-alarm risk) |
| 60 | F29 | Material test-grid generator | v2 (LightBurn already has Material Test) |

---

## 16. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ZoeDepth model download fails on first run | Medium | High | Document offline-cache path; add a `--model-cache` flag. |
| User on CPU-only machine, sliders feel sluggish | High | Medium | Cache depth between previews; show CPU warning. |
| Operator forgets to invert polarity, engraves the inverse | Medium | High | Default to `black_is_deep=True` (LightBurn-correct), show polarity in preview label. |
| 16-bit PNG opened in non-LightBurn tools shows as black | Low | Low | Always write the 8-bit version too; document this. |
| Profile YAML hand-edited with bad values | Medium | Medium | Schema validation with line-numbered errors. |
| Gradio version churn breaks the UI | Medium | Medium | Pin `gradio>=4,<5` in env file; smoke test on upgrade. |

---

## 17. Open Questions

These don't block starting, but should be resolved before v1 ships:

1. **Default model.** `ZoeD_NK` (auto-routes indoor/outdoor) is safest, but `ZoeD_N` is faster and fine for tabletop subjects. *Proposed: default `ZoeD_NK`, expose dropdown.*
2. **Profile authoring UX.** "Save current sliders as new profile" button in v1, or v2? *Proposed: v2.*

---

## 18. Definition of Done (v1)

- `python apps/mopa2lightburn.py --gui` opens the app in the default browser.
- Dropping a 2000×2000 photo, picking `mopa_60w_brass`, clicking Export produces all 4 output files in `outputs/` within 5 seconds on a CUDA machine.
- The exported `_lightburn.png` opens in LightBurn and behaves correctly in 3D Sliced mode (verified by the operator on real metal).
- `pytest` passes on a fresh checkout without GPU.
- README's first 30 lines describe **this app**, not generic depth estimation.

---

## 19. Reference Machine Profile (OMTech MOPA 60W)

Real hardware the app targets first. All defaults, guards, and starting-point profile parameters are tuned for this machine.

| Parameter | Value |
|---|---|
| Laser source | JPT MOPA fiber |
| Output power | 60 W |
| Wavelength | 1064 nm |
| Frequency range | 1–4000 kHz |
| Pulse-width range | 2–500 ns |
| Marking area | 175 × 175 mm (6.9" × 6.9") |
| Max marking speed | 10 000 mm/s |
| Max marking depth | ~0.10 mm (per typical 3D Sliced run) |
| Beam quality (M²) | ≤ 1.4 |
| Positioning accuracy | ±0.1 µm |
| Native software | EzCad2 (we target LightBurn instead) |
| Cooling | Air |

**Implications for the app:**
- **Workpiece canvas defaults to 175 × 175 mm** in workpiece templates and `.lbrn2` exports.
- **Depth-budget UI is bounded 0–100 µm** (with a soft warning above 80 µm).
- **Pulse-width presets** are bucketed into MOPA-meaningful zones: `2 ns` (cold mark / color), `20 ns` (general), `100 ns` (deep), `200 ns` (deepest), `500 ns` (max).
- **Frequency presets** mirror common MOPA tempering recipes for stainless and titanium colors.
- **Min feature-size guard** uses lens-derived spot sizes:

  | Lens | Approx. spot Ø | Recommended min line interval |
  |---|---|---|
  | 70 mm | ~25 µm | 0.020 mm |
  | 110 mm | ~35 µm | 0.025 mm |
  | 150 mm | ~45 µm | 0.030 mm |
  | 200 mm | ~60 µm | 0.040 mm |
  | 300 mm | ~90 µm | 0.060 mm |

A new profile `profiles/machine_omtech_mopa_60w.yaml` captures the immutable hardware envelope; material profiles reference it via a `machine: omtech_mopa_60w` field.

---

## 20. Full Image-Processing Suite

The pipeline expands from "normalize + tone-map" into a true four-stage suite. Every stage is composable and toggleable.

### 20.1 Stage A — Input conditioning (before depth inference)

New module: `mopa/imgproc/input.py`

| Step | Algorithm / library | Notes |
|---|---|---|
| EXIF orientation fix | PIL `ImageOps.exif_transpose` | Always on. |
| Auto white balance | Gray-world or Retinex (`cv2.xphoto`) | Toggle. |
| Exposure normalization | CLAHE on L channel | Strength slider. |
| Shadow lift / highlight recover | Tone-mapped LUT | Two sliders. |
| Specular highlight removal | Threshold + `cv2.inpaint` | Toggle. |
| Denoise | Non-local-means (`cv2.fastNlMeansDenoisingColored`) | Strength slider. |
| Upscale | Real-ESRGAN (optional, weights downloaded on demand) | For low-res sources. |
| Subject auto-mask | `rembg` (default) or SAM (optional) | Output reused by background flatten + multi-pass layers. |
| Perspective rectification | Ellipse fit (`cv2.fitEllipse`) → affine warp | One-click for round subjects. |
| Center + crop to aspect | Pad/crop to match workpiece template | Drives DPI stamp. |
| DPI/mm assignment | Operator types real diameter → PIL pHYs chunk on save | Eliminates LightBurn re-scale. |

### 20.2 Stage B — Depth synthesis

New module: `mopa/imgproc/depth.py`

| Step | Notes |
|---|---|
| ZoeDepth inference (`ZoeD_N` / `K` / `NK`) | Existing. |
| Optional MiDaS pass + ensemble | Median-fuse two depth maps; lifts mid-frequency detail. |
| Bilateral inference TTA | Run on N rotated/zoomed copies, median-fuse. |
| Tiled inference with feathered weights | Required for >4K inputs. |
| Edge-aware refinement | `cv2.ximgproc.jointBilateralFilter` with the RGB image as guide. |
| Frequency separation | Gaussian split into low + high; expose as form/detail sliders. |

### 20.3 Stage C — Heightmap shaping (post-processing)

Existing `mopa/heightmap.py`, expanded:

| Step | New / existing |
|---|---|
| Percentile clipping | existing |
| Polarity remap | existing |
| Gamma / contrast / midtone boost | existing |
| Deep-limit / surface-limit reservation | existing |
| Background flatten (threshold or mask-driven) | upgrade to use subject mask |
| Bilateral / Gaussian smoothing | existing |
| Unsharp-mask sharpen | existing |
| Per-material gray→depth LUT | NEW (F17) |
| Max-depth budget remap | NEW (F23) |
| Floyd–Steinberg dither at 16→8 quantization | NEW (F21) |
| Posterization preview to N passes | NEW (F20) |
| Symmetric/radial cleanup for round subjects | NEW (F24) |
| Min-feature-size guard | NEW (F22, warning only) |

### 20.4 Stage D — Layer derivation (multi-pass)

New module: `mopa/imgproc/layers.py`. See §21.

### 20.5 UI surfacing

The Studio UI gets a left-side stage selector (Input → Depth → Heightmap → Layers → Export). Each stage has its own collapsible panel of controls. A persistent live preview thumbnail strip shows the result of every stage so the operator can see where an artifact entered.

---

## 21. Multi-Pass / Multi-Layer Export Specification

Goal: stop emitting one PNG. Start emitting a **layered job** that uses the full MOPA capability stack — depth, cleanup, color, polish, fiducials — each tagged with its own LightBurn cut settings.

### 21.1 Layer model

Internal representation: a `LayerStack` dataclass.

```python
@dataclass
class Layer:
    id: str                         # "depth", "cleanup_surface", "color_blue", ...
    role: str                       # see table below
    image: np.ndarray | None        # 8- or 16-bit grayscale, or RGBA mask
    geometry: list[Shape] | None    # vector shapes (cuts, fiducials, frames)
    cut_settings: CutSettings       # power/speed/freq/pulse/passes/line_interval/angle/mode
    output_priority: int            # LightBurn layer order; lower runs first
    enabled: bool
    color_index: int                # LightBurn color slot 00-29

@dataclass
class CutSettings:
    mode: str                       # "3D Sliced" | "Image" | "Fill" | "Line" | "Offset Fill"
    power_percent: float
    speed_mm_s: float
    frequency_khz: float
    pulse_width_ns: float
    passes: int
    line_interval_mm: float
    angle_deg: float
    angle_increment_deg: float
    z_offset_mm: float
    bidirectional: bool
    output: bool                    # whether layer runs (vs. preview-only)
```

### 21.2 Layer roles shipped in v1.5

| ID | Role | Source | Default LightBurn settings (60 W MOPA, brass) |
|---|---|---|---|
| `depth` | 3D-Sliced relief | post-processed heightmap | mode=3D Sliced, 92 %, 2000 mm/s, 100 kHz, 200 ns, 256 passes, 0.025 mm, 45° inc |
| `cleanup_surface` | Solid pass over engraved area to deburr | subject mask, dilated 0.2 mm | mode=Fill, 30 %, 4000 mm/s, 200 kHz, 20 ns, 1 pass, 0.04 mm, 90° |
| `background_ablate` | Strip background to uniform texture | inverse subject mask | mode=Fill, 60 %, 3000 mm/s, 60 kHz, 100 ns, 2 passes, 0.03 mm, 0° |
| `shadow_deepen` | Re-pass deepest pockets | mask of bottom 10 % heightmap values | mode=Image, 95 %, 1500 mm/s, 80 kHz, 200 ns, 32 passes, 0.025 mm, 45° |
| `highlight_polish` | Polish brightest peaks for contrast | mask of top 10 % heightmap values | mode=Fill, 25 %, 6000 mm/s, 400 kHz, 4 ns, 1 pass, 0.03 mm, 0° |
| `outline_emboss` | Sharp edges from source | Canny / structured-edge on input | mode=Line, 70 %, 2500 mm/s, 100 kHz, 50 ns, 3 passes |
| `color_*` | Tempered-color zones (stainless/titanium) | per-zone masks | mode=Fill, see §21.4 |
| `fiducial` | Corner crosshairs for re-alignment | procedural | mode=Line, 80 %, 1000 mm/s, 100 kHz, 100 ns, 1 pass |
| `serial_caption` | Auto-generated date/serial text | text engine | mode=Fill, 60 %, 2000 mm/s, 100 kHz, 20 ns, 1 pass |
| `art_board_preview` | Composite color-coded preview only | render | output=false |

### 21.3 Additional roles for v2/v3

`black_mark`, `frosted_white`, `stipple`, `cross_hatch`, `edge_bevel`, `coating_burn_off`, `pre_burn_ablate`, `qc_ramp`, `watermark`, `jig_outline`, `microtext`, `thermal_idle`, `negative_relief_mate`, `adaptive_line_interval`, `anti_alias_outline`, `finger_grip_texture`, `selective_color_mask_pack`. Schema is the same; only the source-mask generator differs.

### 21.4 Color tempering presets (MOPA on stainless)

Profile carries a `color_recipes` block:

```yaml
color_recipes:
  stainless:
    deep_blue:    { freq_khz: 60,  pulse_ns: 4,   speed: 800,  power: 35, line_interval: 0.04, passes: 1 }
    royal_blue:   { freq_khz: 80,  pulse_ns: 4,   speed: 1000, power: 30, line_interval: 0.04, passes: 1 }
    purple:       { freq_khz: 100, pulse_ns: 8,   speed: 1200, power: 28, line_interval: 0.04, passes: 1 }
    gold:         { freq_khz: 200, pulse_ns: 20,  speed: 1500, power: 25, line_interval: 0.04, passes: 1 }
    bronze:       { freq_khz: 150, pulse_ns: 50,  speed: 1300, power: 32, line_interval: 0.04, passes: 1 }
    black:        { freq_khz: 20,  pulse_ns: 200, speed: 500,  power: 80, line_interval: 0.025, passes: 2 }
```

Every color name a recipe supports becomes a paint color in the ROI brush (v3); for v1.5 the operator selects one color and one zone-mask source.

### 21.5 LayerStack → disk

Default export now writes:

```
outputs/<stem>/
  ├─ layers/
  │  ├─ 01_depth.png
  │  ├─ 02_cleanup_surface.png
  │  ├─ 03_background_ablate.png
  │  ├─ 04_shadow_deepen.png
  │  ├─ 05_highlight_polish.png
  │  └─ ...
  ├─ vectors/
  │  ├─ fiducials.svg
  │  └─ outline.svg
  ├─ <stem>_art_board.png         # color-coded preview of all layers
  ├─ <stem>.lbrn2                 # see §22
  ├─ <stem>.clb                   # material library with this profile
  ├─ <stem>_settings.json
  └─ <stem>_master16.png          # legacy single-layer master
```

For users who still want the v1 single-PNG flow, a setting `export.layered = false` collapses the stack to one `_lightburn.png`.

---

## 22. LightBurn-Native Export (.lbrn2 / .clb / .lbset)

**Yes, the app can produce files LightBurn imports directly.** LightBurn's project and library formats are XML, well-documented enough to author programmatically without reverse engineering.

### 22.1 What we generate

| File | Format | Purpose | Phase |
|---|---|---|---|
| `<stem>.lbrn2` | LightBurn 2 project (gzipped or plain XML) | Open and run a job in one double-click. | v2 (Phase 4) |
| `<stem>.clb` | LightBurn Cut Library | Import a named set of cut presets per material. | v1.5 (Phase 3) |
| `<stem>.lbset` | Single-cut settings | Drop on an existing layer to apply one preset. | v1.5 (Phase 3) |
| `<stem>.lbart` | Art library (rare) | Reusable shape library. | v3 |

### 22.2 `.lbrn2` contents we author

- `<Thumbnail>` rendered from the art-board preview.
- `<VariableText>` block populated with profile name, material, date, serial.
- `<MachineSettings>` workspace size = 175 × 175 mm (per §19).
- `<CutSetting>` blocks: one per layer in the LayerStack, on its own LightBurn color slot (00–29). Each carries `power`, `maxPower`, `speed`, `frequency`, `qPulseWidth`, `numPasses`, `interval`, `angle`, `bidir`, `tabsEnabled=False`, `output=True/False`, `priority`.
- `<Shape>` blocks per layer:
  - **Image layers** (`depth`, `cleanup_surface`, `background_ablate`, `shadow_deepen`, `highlight_polish`, color masks): `<Shape Type="Bitmap">` with embedded base64 image data, `<ImageProcessingMode>` = `3DSliced` / `Threshold` / `Stucki` / `Greyscale` as appropriate, plus `<DPI>` set from the pHYs stamp so physical size is exact.
  - **Vector layers** (fiducials, outline, microtext): `<Shape Type="Path">` blocks built from the SVG geometry.
- `<Notes>` block embedding the same JSON written to `<stem>_settings.json` so the project is self-documenting inside LightBurn.

### 22.3 `.clb` contents

One `.clb` per material profile, containing every CutSetting from §21.2 plus all color recipes from §21.4. Importing the library populates LightBurn's Material Library panel with entries like:

```
OMTech MOPA 60W › Brass › Depth pass
OMTech MOPA 60W › Brass › Cleanup surface
OMTech MOPA 60W › Stainless › Color: Royal Blue
OMTech MOPA 60W › Stainless › Color: Gold
...
```

This is the single biggest workflow win: operators stop copy-pasting cut settings from forum posts.

### 22.4 Implementation notes

- New module: `mopa/lightburn/` with `lbrn2_writer.py`, `clb_writer.py`, `lbset_writer.py`, and `xml_helpers.py`.
- LightBurn writes `.lbrn2` either as plain XML or gzip-compressed; we'll author plain XML (LightBurn imports both) to keep diffs reviewable.
- DPI handling: LightBurn respects the PNG pHYs chunk *and* the `<DPI>` element. We set both consistently from the operator's mm assignment.
- Compatibility: target LightBurn 1.7+ schema; smoke-test against current public release before each app release.
- Round-trip test: every `.lbrn2` we write is opened, validated, and re-saved by a headless XML schema check in CI.

### 22.5 What we cannot generate

- Live machine connection / send-to-laser. Operator still presses Start in LightBurn.
- Galvo correction files (`.cor`). Hardware-specific, lives on the controller.
- EzCad2 `.ezd` files (we explicitly target LightBurn instead).

---

## 23. Final File & Folder Layout (post v1.5)

```
mopa-heightmap/
├─ apps/
│  └─ mopa2lightburn.py            # CLI + --gui
├─ ui/
│  ├─ app.py
│  └─ mopa_studio.py
├─ zoedepth/
│  └─ laser/
│     ├─ __init__.py
│     ├─ service.py                # orchestrator
│     ├─ exporter.py               # bundle writer (atomic, naming modes)
│     ├─ heightmap.py              # Stage C
│     ├─ preview.py                # shaded relief, histogram, false-color, art-board
│     ├─ profiles.py               # YAML loader + schema validator
│     ├─ tiling.py                 # feathered tiled inference
│     ├─ masks.py                  # subject + background utilities
│     ├─ calibration.py            # ramp generator + LUT fitter
│     ├─ lut.py                    # gray->depth LUT apply/store
│     ├─ layers.py                 # LayerStack + role generators (§21)
│     ├─ color_recipes.py          # MOPA tempering presets
│     ├─ imgproc/
│     │  ├─ input.py               # Stage A: WB, denoise, mask, rectify, DPI
│     │  ├─ depth.py               # Stage B: ensemble, TTA, edge-refine, freq-sep
│     │  └─ layers.py              # Stage D: per-role mask & image generators
│     └─ lightburn/
│        ├─ lbrn2_writer.py        # .lbrn2 author
│        ├─ clb_writer.py          # .clb author
│        ├─ lbset_writer.py        # .lbset author
│        └─ xml_helpers.py
├─ profiles/
│  ├─ machine_omtech_mopa_60w.yaml
│  ├─ mopa_60w_brass.yaml
│  ├─ mopa_60w_stainless.yaml
│  ├─ mopa_60w_aluminum.yaml
│  └─ mopa_60w_copper.yaml
├─ templates/                       # workpiece templates (mm-accurate)
│  ├─ coin_25mm.yaml
│  ├─ dog_tag_50x28.yaml
│  ├─ zippo_lid.yaml
│  └─ knife_scale.yaml
├─ outputs/                         # gitignored
├─ inbox/  outbox/                  # watch-folder
├─ docs/
│  └─ PLAN.md
└─ tests/
```

---

## 24. Phased Implementation Roadmap

Phases are sized so each one is independently shippable.

### Phase 0 — Foundation (✅ done)

- `mopa` package: heightmap, preview, profiles, masks, tiling.
- `apps/mopa2lightburn.py` CLI.
- Initial 4 MOPA YAML profiles.

### Phase 1 — v1 Studio (§14 plus the v1 add-ons from §15)

1. Service layer (`service.py`).
2. Exporter (`exporter.py`) with atomic writes + 3 naming modes + `keep_history`.
3. Profile schema validation.
4. App settings file (`~/.mopa-heightmap/settings.json`).
5. Pytest suite (heightmap, profiles, exporter).
6. Gradio UI (`ui/mopa_studio.py`).
7. CLI `--gui`.
8. Depth cache (input-hash → array).
9. v1 add-ons: EXIF auto-rotate (F8), "Reset to profile defaults" (F51).
10. README rewrite.

Exit criteria: §18 Definition of Done.

### Phase 1.5 — Packaging & Shareability

Lift the project from "developer checkout" to "tech-savvy maker can install in 10 minutes" without waiting for the full PyInstaller exe in Phase 6. Cheap wins that meaningfully lower the install bar.

1. **`pyproject.toml`** with PEP 621 metadata, hatchling build backend, and dependency groups (`core`, `ui`, `dev`). Replaces the conda-only install path.
2. **Console-script entrypoints** so install gives the user real commands on PATH:
   - `mopa-studio` → launches the Gradio UI.
   - `mopa-heightmap` → runs the CLI.
3. **User-scope profiles directory**: `~/.mopa-heightmap/profiles/` is searched first, then `<repo>/profiles/`. Drag-and-drop a YAML there to install it; no need to touch the repo.
4. **CPU-only install path** documented (`pip install mopa-heightmap[ui]` with `torch` resolved from the CPU index URL) so a shop without a GPU can at least try it.
5. **Operator-targeted README** rewrite — "Quick start" first, screenshots, three commands, zero jargon.
6. **`.gitignore` + `MANIFEST.in`** hardening so `outputs/`, `~/.mopa-heightmap/`, model caches stay out of source control and packages don't ship junk.
7. **First-run weight pre-download** button in the UI ("Download model weights now") with progress + cache-location display.
8. **Single zip release on GitHub** with `profiles/`, sample images, and a one-page setup PDF.

Exit criteria:
- `pip install -e .[ui]` from a clean Python 3.10+ venv produces working `mopa-studio` and `mopa-heightmap` commands.
- `mopa-heightmap --make-ramp ramp.png` succeeds with no profile/repo path knowledge.
- A user can drop `my_brass.yaml` into `~/.mopa-heightmap/profiles/` and the UI lists it next to the shipped profiles.
- `pytest -q` passes on a fresh CPU-only install.

Defers to Phase 6: the actual PyInstaller `.exe`, Docker image, and crash-report bundle.

### Phase 2 — v1.5a Image-Processing Suite (Stage A + Stage C upgrades)

Folds in: F1, F3, F5, F6, F7, F14, F21, F22, F23, F32, F33, F34, F42, F46, F47, F48, F52, F54.

1. New `imgproc/input.py` module (WB, denoise, mask, rectify, DPI stamp).
2. Subject auto-mask via `rembg` (GPU optional).
3. Workpiece templates folder + center/crop.
4. Edge-aware joint-bilateral refinement.
5. Floyd–Steinberg dither at 16→8 quantization.
6. Max-depth-budget UI (µm).
7. Min-feature-size guard.
8. Engraving-time estimator.
9. UI: live debounced preview, tooltips, histogram overlays, false-color toggle, side-by-side panel.
10. Batch mode (CLI + UI).
11. Recipe history JSONL.
12. Re-run from `_settings.json` (drop-on-app).

Exit criteria: an operator can take 10 photos, batch-process with one profile, and every export reproduces from its sidecar JSON byte-for-byte.

### Phase 3 — v1.5b Calibration + Multi-Pass + LightBurn Library Export

Folds in: F17, F18, F19, F20, F28, F38, F39, plus §21 multi-pass v1.5 stack and §22.3/4 `.clb` + `.lbset` writers.

1. `lut.py` — monotonic spline fit, gray→depth LUT, applied after tone curve.
2. LUT input UI: 11 measurement boxes next to ramp preview, fit + save into profile.
3. Posterization preview (N passes).
4. Pulse-width-aware LUT slots in profile YAML.
5. `LayerStack` data model + `imgproc/layers.py` role generators (depth, cleanup_surface, background_ablate, shadow_deepen, highlight_polish, outline_emboss, color, fiducial, serial_caption, art_board_preview).
6. Layered export to `outputs/<stem>/layers/`.
7. `.clb` and `.lbset` writers (§22).
8. Color tempering recipe library + UI for choosing one color/zone in v1.5.
9. Setting `export.layered` toggle to keep v1 single-PNG flow.

Exit criteria: importing the generated `.clb` into LightBurn populates the Material Library; layered export opens cleanly in LightBurn one image at a time.

### Phase 4 — v2a `.lbrn2` Project Export

Folds in: F25, F26, F4, F12.

1. `lightburn/lbrn2_writer.py` — author `.lbrn2` from `LayerStack`.
2. CutSetting blocks per layer with full MOPA params (incl. qPulseWidth).
3. Embedded base64 image data + correct DPI from pHYs.
4. `<MachineSettings>` workspace = 175 × 175 mm (or per machine profile).
5. `<Notes>` JSON for self-documentation.
6. Workpiece templates wired into project canvas.
7. Perspective rectification (ellipse → circle).
8. MiDaS + ZoeDepth ensemble.
9. CI round-trip schema test.

Exit criteria: double-clicking a generated `.lbrn2` opens a fully-configured LightBurn job that runs without edits.

### Phase 5 — v2b Depth Quality + UX Polish

Folds in: F2, F11, F15, F16, F24, F30, F31, F35, F36, F45, F50, F53, F55.

1. Specular highlight removal.
2. Bilateral inference TTA.
3. Frequency separation sliders.
4. Feathered tile blending; expose tiling in UI.
5. Symmetric/radial cleanup for round subjects.
6. Multi-lens calibration (LUT per lens).
7. Watch folder.
8. Undo/redo.
9. A/B compare panel.
10. Reflective-surface warning.
11. Pin/lock sliders across profile changes.
12. 3D mesh preview tab.
13. Drag-and-drop profile YAML install.

Exit criteria: app feels finished. Power users have everything they need.

### Phase 6 — v3 Power-User + Distribution

Folds in: F9, F37, F41, F44, F49, F56, F57, F59, F60, plus the v2/v3 multi-pass roles in §21.3.

1. Region-of-interest brush (selective flatten/sharpen/protect, paint color zones).
2. Multi-image fusion, per-batch calibration metadata, bookmarks.
3. Additional layer roles: `black_mark`, `frosted_white`, `stipple`, `cross_hatch`, `edge_bevel`, `coating_burn_off`, `pre_burn_ablate`, `qc_ramp`, `watermark`, `jig_outline`, `microtext`, `thermal_idle`, `negative_relief_mate`, `adaptive_line_interval`, `anti_alias_outline`, `finger_grip_texture`, `selective_color_mask_pack`.
4. PyInstaller one-file build (Windows-first).
5. Dockerfile (CUDA base) for shop standardization.
6. Local model-cache management UI.
7. Crash-report bundle.

Exit criteria: app distributable to a non-developer's shop PC; full multi-pass spectrum exposed.

---

## 25. Cross-Phase Dependency Graph

```
Phase 1 (Studio)
  ├─ service.py ─────────────────────┐
  ├─ exporter.py ────────────────────┤
  └─ settings.json ──────────────────┤
                                     │
Phase 2 (Image suite) · needs service+exporter
  ├─ imgproc/input.py ───────────────┤
  ├─ subject mask (rembg) ──────────┐│
  ├─ templates/ + DPI stamp         ││
  └─ history JSONL + re-run         ││
                                    ││
Phase 3 (Calibration + Multi-pass + .clb)
  ├─ lut.py ────────────────────────┬┘│
  ├─ LayerStack ←───────────────────┤  │
  ├─ color recipes                  │  │
  └─ .clb / .lbset writers ← LayerStack│
                                       │
Phase 4 (.lbrn2) · needs LayerStack + LUT
  ├─ lbrn2_writer.py
  ├─ perspective rectify
  └─ MiDaS+Zoe ensemble

Phase 5 (Polish) · mostly independent
Phase 6 (Power user) · needs ROI brush ← mask infra from Phase 2
```

---

## 26. Updated Definition of Done

### v1 (Phase 1)
- See §18.

### v1.5 (Phases 2 + 3)
- Operator runs the calibration ramp once, types 11 measured depths, and every subsequent export on that material lands within ±5 µm of the requested depth budget.
- Layered export of a coin photo produces depth + cleanup + background-ablate + highlight + shadow + fiducials + art-board, plus a `.clb` library that imports cleanly into LightBurn.
- Batch processing 50 images with one profile completes without manual intervention and writes a recipe-history line per export.

### v2 (Phases 4 + 5)
- One double-click on a generated `.lbrn2` opens LightBurn with the workpiece sized correctly, all layers on the right colors with the right MOPA settings, and the operator can press Start without further edits.
- A/B compare and frequency-separation sliders make tuning a new material take ≤30 minutes.

### v3 (Phase 6)
- Region-of-interest brush lets the operator paint "this part stays raised, this part gets blue tempering, this part gets black mark" and the layer stack updates live.
- A non-developer can install the app on a shop Windows PC from a single executable and run their first job in under 10 minutes.
