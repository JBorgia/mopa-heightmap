# MOPA Heightmap Studio

Turn a photo into a **LightBurn 3D Sliced** engraving bundle, tuned for
**JPT MOPA fiber galvo** lasers.

The pipeline is sculptok-first:

```
photo  →  pre-sculptok prep  →  sculptok  →  bundle writer  →  LightBurn
         (CLAHE/denoise/specular)            (3D-Sliced .lbrn2 + .clb +
                                              per-pass PNGs + mask + STL)
```

[Sculptok](https://www.sculptok.com/) generates the depth heightmap.
This project supplies everything else: pre-sculptok image conditioning,
opt-in refinement passes (subject mask, photo-tonal overlay, signature),
target-object presets (coin / signet ring / pendant / plaque / portrait),
material-card-driven cut settings, and a full LightBurn bundle that
opens in 3D Slice mode without manual setup.

> **Status (May 2026):** sculptok-only since the v10 rewrite. The local
> depth-inference machinery (ZoeDepth / DepthAnything-V2 / face_relief
> / TripoSR / SDXL stylizer / Real-ESRGAN x4) is gone. See
> `memory/project_sculptok_only_pivot.md` for the rationale.

---

## Quick start

### 1. Install Python deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[mask,dev]"
```

The `[mask]` extra adds `rembg` + `onnxruntime` for the subject-mask
deliverable (still CPU-only; no CUDA build needed).

### 2. Configure a Sculptok API key (optional but recommended)

Add it to `~/.mopa-heightmap/settings.json`:

```json
{
  "credentials": {
    "sculptok_api_key": "YOUR_KEY"
  }
}
```

Or set `SCULPTOK_API_KEY` in the environment, or pass `--sculptok-api-key`
on the CLI. Without a key you can still bring your own heightmap PNG via
`--heightmap`.

### 3. Run the CLI

```powershell
mopa-heightmap photo.jpg --use-sculptok --target portrait `
    --output outputs --export-lbrn2 --export-clb --export-mask
```

Or skip Sculptok and bring your own heightmap PNG:

```powershell
mopa-heightmap photo.jpg --heightmap photo_relief.png `
    --output outputs --target signet_ring --export-lbrn2
```

The bundle lands in `outputs/<name>/`:

```
final/                  ← drag this folder into LightBurn
  project.lbrn2         (3D-Sliced layer + opt-in refinement layers)
  cut-library.clb
  lightburn.png         (8-bit master)
  master16.png          (16-bit master, byte-identical to sculptok)
  pass_NN_<kind>.png    (per-pass layer PNGs, embedded in the .lbrn2 too)
  mask.png              (subject silhouette deliverable, when --export-mask)
work/
  preview.png           (shaded relief)
  settings.json         (everything reproduces this run)
```

Drop `final/project.lbrn2` into LightBurn 1.7+. The C01 layer is
pre-set to **3D Slice** mode at 256 passes — adjust passes / power /
speed to match your material before firing.

---

## Web UI

Two processes, one command. From `apps/web/`:

```powershell
pnpm install      # first time only
pnpm start        # API on :8000, web on :4200, both with reload
```

`pnpm start` runs both in parallel via `concurrently`; Ctrl+C kills the
pair. CORS is wired so the web at `localhost:4200` can talk to the API
at `127.0.0.1:8000` directly.

The Studio shell exposes:

- **Mask** — backend (BiRefNet / RemBG / Threshold) + edge softness +
  click-refine.
- **Pre-sculptok input prep** — White balance / CLAHE / Denoise /
  Specular removal / Max input dim.
- **Render** — Material profile dropdown + Render button.
- **Heightmap** — Target-object preset dropdown, "Generate via
  Sculptok" button (with credit balance), source polarity, polarity
  invert, black is deep, background value.
- **Refinement passes** — Subject mask deliverable, Pre-clean,
  Photo-tonal overlay (with strength + invert), Signature text + corner,
  Output dither.
- **Output** — heightmap blob, Compute pass plan, Export PNG / .lbrn2 /
  .stl.

The Wizard route walks the same controls in 5 pages: Upload → Subject →
Prep & Refine → Material & Passes → Review & Export.

---

## CLI flags reference

```
mopa-heightmap <input.jpg> [<input2.jpg> ...] [options]

Heightmap source (one required):
  --heightmap <path>           Bring-your-own heightmap PNG
  --use-sculptok               Auto-pull from sculptok (consumes credits)
    --sculptok-api-key <key>   Override env var / settings.json
    --sculptok-style {normal|portrait|sketch|pro}        (default: pro)
    --sculptok-version {1.0|1.5}                          (default: 1.5)
    --sculptok-hd {2k|4k}                                 (default: 2k)

Target / material:
  --target <name|path>         Coin / signet_ring / pendant / plaque / portrait
  --profile <name|path>        Material profile (mopa_60w_stainless / ...)
  --lightburn-card <name>      Override the cut-card (default: Colour60W-M7)

Heightmap polarity:
  --heightmap-polarity {bright_raised|dark_raised|auto}
  --polarity-invert            Signet ring / recessed mode

Pre-sculptok input prep:
  --white-balance / --clahe / --denoise / --remove-specular
  --clahe-clip <f> / --denoise-strength <f> / --specular-threshold <i>
  --max-input-dim <px>

Refinement passes:
  --subject-mask               Compute + ship subject mask deliverable
  --subject-mask-backend {rembg|birefnet|threshold}
  --photo-tonal                Photo-luminance overlay pass
    --photo-tonal-strength <0..1> / --photo-tonal-invert
  --signature "TEXT"           Corner signature
    --signature-corner {tl|tr|bl|br}

Output:
  --output <dir>               Parent directory (default ./outputs/)
  --name <stem>                Bundle folder name
  --export-preview             Shaded preview PNG
  --export-calibration-ramp    Calibration ramp PNG
  --export-lbrn2               LightBurn project + per-pass PNGs
  --export-clb                 Cut Library (.clb)
  --export-mask                Subject mask PNG sibling deliverable
  --print-width-mm <f> / --print-height-mm <f>
  --naming {overwrite|timestamp|counter}
  --keep-history               Never overwrite previous exports
```

---

## Repo layout

```
mopa/                  Pipeline package (renamed from zoedepth/laser).
                       service.py orchestrates render+export. sculptok_client.py
                       wraps the Sculptok REST API. lbrn_writer.py builds
                       project.lbrn2 with embedded base64 bitmaps + 3D-Slice
                       cut settings.
apps/api/              FastAPI server. routes/{render,sculptok,targets,
                       mask,plan,export,profile,session,upload,blob}.py.
                       schemas.py mirrors HeightmapSettings 1:1.
apps/web/              Angular 21 frontend. pnpm-managed.
apps/mopa2lightburn.py CLI entry point (mapped to `mopa-heightmap` script).
profiles/              Material cards (.yaml) + targets/ (target-object presets)
LightBurn Colour Card/ Source LightBurn cut-setting libraries (M7 cards).
tests/                 Backend pytest suite (266 passing).
memory/                Project memory for future agent sessions — keep MEMORY.md
                       up to date so the next agent session has the right
                       architectural context.
docs/                  Implementation plan, UI migration brief, and
                       LIGHTBURN_FILE_FORMAT.md reference.
```

---

## Sculptok API key resolution

Lookup chain, first hit wins:

1. CLI flag `--sculptok-api-key`
2. `SCULPTOK_API_KEY` environment variable
3. `~/.mopa-heightmap/settings.json` → `credentials.sculptok_api_key`

The web UI uses option 2 or 3 (server-side resolution). The
`/sculptok/credits` endpoint reports `configured: false` when none of
those are set, and the "Generate via Sculptok" button stays disabled.

---

## Material profiles

The repo ships starter material profiles in `profiles/`:

- `mopa_60w_stainless.yaml`, `mopa_60w_brass.yaml`,
  `mopa_60w_aluminum.yaml`, `mopa_60w_copper.yaml` — 60W M7 starting
  points for the four metals most engravers care about.
- `sculptok_portrait.yaml` — pre-sculptok prep recipe tuned for
  portraits (CLAHE + denoise + specular removal, subject mask on).

Cut-setting parameters are lifted **verbatim** from the LightBurn
material cards in `LightBurn Colour Card/`. Use the [LightBurn Material
Test grid](https://docs.lightburnsoftware.com/latest/Reference/MaterialTest/)
to calibrate your specific machine + material before relying on the
defaults.

---

## Target-object presets

`--target <name>` layers in shape-specific defaults (print dimensions,
polarity invert, starter heightmap-settings block):

| Preset | Print size | Polarity | Notes |
|---|---|---|---|
| `coin` | 50 × 50 mm | normal | Round coin / medal |
| `signet_ring` | 30 × 25 mm | **inverted** | Recessed (intaglio) design |
| `pendant` | 40 × 50 mm | normal | Vertical pendant |
| `plaque` | 80 × 60 mm | normal | Rectangular, full-frame subject |
| `portrait` | 50 × 65 mm | normal | Heavy pre-sculptok prep |

Override any of them with `--target <path-to-yaml>` to ship a custom
preset.

---

## Tests

Backend:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Frontend (from `apps/web/`):

```powershell
pnpm test            # one-shot, CI-mode (ng test --watch=false under vitest)
pnpm test:watch      # interactive
```

Both backend (266 tests) and frontend (120 tests) suites are wired into
the standard runners; no special setup needed beyond `pnpm install`
and `pip install -e ".[mask,dev]"`.

---

## Architecture memory

The `memory/` directory carries project-memory files that future agent
sessions read on startup. The current pinned facts:

- `project_sculptok_only_pivot.md` — depth pipeline removed 2026-05-05;
  do not reintroduce DAv2 / face_relief / Real-ESRGAN / etc.
- `project_pass_architecture.md` — depth layer is the sculptok PNG
  itself; refinement passes ADD features on top, never subdivide the
  depth budget. Burn-through bug from the old band-slicer approach is
  documented here.
- `feedback_mask_handles_subject_isolation_in_lightburn.md` — sculptok
  output fills the frame; the mask is a separate deliverable LightBurn
  applies at engrave time.
- `reference_node_toolchain.md` — pnpm + Angular CLI install paths on
  the dev box.

---

## License

MIT for this project's source. Sculptok output and LightBurn material
cards retain their respective upstream licenses; bundle outputs depend
on which heightmap source the user supplied.
