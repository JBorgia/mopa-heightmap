---
name: open backlog — features that need their own focused session
description: As of 2026-05-06, three big features from the original ranked list (material profile persistence, mediapipe-driven auto-orient/auto-crop, procedural background generators) are intentionally deferred. Each is large enough to warrant a dedicated session with its own design + test pass.
type: project
---

The plan-completion sweep on 2026-05-06 landed sculptok auto-pull,
target-object presets, Vitest, and a README rewrite. Three items from
the original ranked feature list are deferred:

## 1. Material + laser specs persistence (was #2 on the rank)

**Current state:** Cut settings are lifted verbatim from the LightBurn
material cards in ``LightBurn Colour Card/`` (Colour20W-M7, 30W, 60W,
80W, 100W). The CLI accepts ``--lightburn-card <name>`` to pick which
card, and ``--profile <name>`` for a starter heightmap-settings YAML.

**What's still missing:**
- A server-side ``MaterialProfile`` schema with
  ``(power, speed, freq, Q-pulse, passes) → anneal_color`` rows so the
  per-color-cluster pass picks the right LightBurn settings without
  hand-editing.
- User-account persistence (a DB layer; SQLite + SQLAlchemy is the
  obvious starting point) so each user's calibrated profiles survive
  across sessions.
- A "Material Test grid" generator (see commit log; was a high-rank
  item in the brainstorm). The MOPA community standard is to sweep
  two parameters and record swatches; we should emit those grids as
  bundles directly.

**Why deferred:** Schema design + DB choice + user-auth surface is its
own roadmap. Doing it in the same session as the Vitest fix and README
rewrite would conflate scope.

## 2. Auto-orient face + auto-crop (was #5 / #6)

**Current state:** Pre-sculptok input prep currently does CLAHE,
denoise, white-balance, specular removal, and max-dim resize. Face
orientation and saliency-aware cropping are NOT wired.

**What's still missing:**
- Add ``mediapipe`` as an optional dependency (in the
  ``[mask]`` extras-block alongside ``rembg``).
- Implement ``auto_orient_face(image)`` using MediaPipe FaceMesh:
  rotate so the inter-pupillary line is level. No-op when no face
  detected. Lives in ``mopa/imgproc/auto_orient.py``.
- Implement target-aware auto-crop: if ``--target portrait`` is set,
  centre-crop on the face landmarks. If ``--target coin``, use the
  saliency-aware crop (``cv2.saliency.StaticSaliencySpectralResidual``)
  to find the subject and pad to the target's aspect ratio.
- Wire ``input_auto_orient_face: bool`` and ``input_auto_crop: bool``
  fields on ``HeightmapSettings``; the CLI gets ``--auto-orient`` and
  ``--auto-crop`` flags, the UI gets two checkboxes in the Pre-sculptok
  Input Prep panel.

**Why deferred:** MediaPipe install on Windows can be flaky (model
download, ABI mismatches). Worth its own session that includes a
careful ``pip install`` smoke + a sample-image regression test.

## 3. Procedural background generators (was the BG-replace flow)

**Current state:** No procedural background generation. Users supply
a finished sculptok PNG (which fills the frame) and the bundle ships
that as-is.

**What's still missing:**
- New module ``mopa/backgrounds/`` with procedural generators:
  - ``guilloche.py`` — engine-turned curves (Lissajous-style
    polar-coordinate sin/cos compositions).
  - ``stripes.py`` — straight / diagonal hatch patterns at a given angle.
  - ``dots.py`` — Bridson Poisson-disk sampling for organic stipple.
  - ``halftone.py`` — radius modulated by an underlying tone source.
  - ``checkers.py`` — basic two-cell pattern.
- A pre-sculptok pipeline stage: when the photo's background is
  flat (corner-sample > 0.95 or BiRefNet alpha < 0.2), composite the
  procedural background on top of the matted-out background BEFORE
  feeding sculptok. Sculptok then renders the full frame as relief.
- UI: new "Background" panel in the Studio (between Mask and Pre-sculptok
  Input Prep) with: pattern dropdown, size slider, angle slider,
  custom-image upload, "Use original" passthrough.

**Why deferred:** Real procedural-art design work — each pattern is
50–150 lines, plus a UI panel, plus a regression-test fixture. Best
done as a single focused session so the patterns share a common
sampling / dithering convention.

## Reasonable next-session ordering

1. **Material profile schema** (no DB yet — just a schema + a JSON
   blob in ``~/.mopa-heightmap/profiles/<name>.json``). Smallest of the
   three; can ship without a DB layer.
2. **Auto-orient face** (one stage in the pipeline; matches the existing
   CLAHE / denoise pattern).
3. **Background generators** (after auto-orient lands so the BG can be
   composited against an oriented subject).

## Items already covered (so the next agent doesn't re-do them)

Don't re-litigate these — they shipped this session:

- ✅ Sculptok auto-pull (UI + backend) — commit ``<sculptok+targets>``
- ✅ Target-object presets — same commit
- ✅ Vitest setup fixed — commit ``<vitest>``
- ✅ README rewrite — commit ``<readme>``
- ✅ All HeightmapSettings fields exposed in the Studio UI
- ✅ Wizard page 2 ("Prep & Refine") wired to the same state
