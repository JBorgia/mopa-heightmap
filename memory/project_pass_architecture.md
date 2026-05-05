---
name: pass architecture — sculptok is the depth cut, refinement layers add features
description: The .lbrn2 has ONE 3D Sliced bitmap layer (sculptok heightmap, untouched) plus optional refinement layers that add separate physical features (hair detail, eye lines, color zones, shading, signature, cut outline). Refinement passes do NOT subdivide the heightmap — slicing the heightmap into frequency bands compounds the engraving depth and burns through the part.
type: project
---

The user articulated this on 2026-05-05 after the 8-pass smoke test
revealed my form/cleanup/detail/shading/polish decomposition was
slicing the same heightmap into bands. With each band asking for
full-depth at overlapping pixels, LightBurn would fire a cumulative
~3-5× the design depth — burning through the part.

**Correct architecture:**

```
.lbrn2 layers (one CutSetting per layer):

  [optional] pre_clean    : defocused full-frame, oxide/oil burn-off
  [REQUIRED] depth        : sculptok PNG, Mode="3DSliced"  ← THE depth cut
  [optional] photo_detail : photo-derived hair / beard / fur / fabric edges
                            (Sobel/Laplacian on the photo, gated by mask)
  [optional] eye_lines    : tight raster or vector on eye region
  [optional] color:CXX    : LAB k-means clusters, MOPA anneal power
  [optional] photo_tonal  : low-power dithered photo-luma overlay
  [optional] shading      : photo-derived tonal darkening
  [optional] signature    : vector text in a corner
  [optional] frame/border : vector decoration
  [optional] cut_outline  : vector cut-through line
```

**Why:** sculptok produces an engraving-ready heightmap. The 3D Sliced
cut on that PNG carries the entire depth budget. Refinement layers are
**not depth subdivisions** — they're separate physical operations
(anneal color, low-power shading, vector cuts) that the laser fires on
top of the carved relief. Each refinement layer must use cut settings
that DO NOT carve depth (anneal-only, low power, or vector). They are
additive features, not finer slicing of the same Z budget.

**How to apply:**

- **Never** slice the sculptok heightmap into frequency bands and emit
  them as raster layers. The form/cleanup/detail/shading/polish
  decomposition is dead — `derive_pass_masks` and any planner code
  that produces those should be removed or repurposed.
- The depth pass is **one Bitmap shape** with the sculptok PNG as
  ``SourceFile`` and Mode="3DSliced". No mask, no derivation.
- Refinement passes are **photo-derived** (or vector-derived), not
  heightmap-derived. Hair detail comes from a Sobel on the photo, not
  from `|h - blur(h)|`.
- Each refinement layer has its own cut setting and its own mode
  (Mode="Image" for raster shading, Mode="Fill"/"Line" for vectors).
  None of them should ask LightBurn to carve depth — that's the
  sculptok layer's job exclusively.
- Pre-sculptok prep (CLAHE, denoise, white-balance, specular removal,
  BG isolation) is the OTHER place we add value. Anything that makes
  sculptok's output cleaner.
- Two places we add value: pre-sculptok image prep, and post-sculptok
  refinement features. Nowhere else.
