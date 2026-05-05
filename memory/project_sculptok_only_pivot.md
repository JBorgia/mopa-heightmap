---
name: sculptok is the only heightmap source — depth pipeline removed
description: As of 2026-05-05, MOPA Heightmap Studio uses sculptok exclusively for heightmap generation. The local depth-inference machinery (ZoeDepth, DAv2, Sapiens, TripoSR, Real-ESRGAN, face_relief, depth_unsharp, ControlNet stylizer) has been deleted from the codebase. Don't reintroduce it.
type: project
---

The product is now: photo + sculptok PNG -> LightBurn 3D Sliced bundle.
We don't generate depth locally; we wrap sculptok output into engraving
deliverables (.lbrn2 + .clb + per-pass PNGs + subject mask + .clb + ...).

**Why:** the local-depth track plateaued at ~80-85 % of sculptok's
quality on a 4 GB Quadro P2000 even after stacking DAv2_Base +
Real-ESRGAN x4 + ControlNet-bas-relief + face_relief. Closing the gap
needed a fine-tuned bas-relief LoRA we don't have. The user explicitly
said "you can't do better than sculptok" and pivoted to using their
output. Three commits on 2026-05-05 (1613bb9, 3f5dc1b, f328f53) ripped
the entire depth pipeline (~63k lines).

**How to apply:**
- Don't suggest adding back ZoeDepth, DAv2, Sapiens, TripoSR, Marigold,
  Real-ESRGAN, ControlNet, face_relief, depth_unsharp, or any depth-
  domain processing. The experiment concluded.
- ``service.render()`` requires ``settings["external_heightmap_path"]``
  (sculptok auto-pull or user-supplied PNG). It will raise otherwise.
- Sculptok output is engraving-ready — DO NOT mutate it (no auto-stretch,
  no tone curve, no smoothing, no masking-into-the-heightmap). Polarity
  invert (signet-ring mode) is the only allowed transform.
- The subject mask is a SEPARATE deliverable artifact (mask.png), not
  applied to the heightmap. LightBurn handles subject/background
  isolation at engrave time.
- The package directory ``zoedepth/laser/`` is misleadingly named (we
  forked from the ZoeDepth repo) but a rename is deferred — don't
  re-link to upstream ZoeDepth or assume any depth code lives there.
- Pre-sculptok image conditioning (CLAHE, denoise, white-balance,
  specular removal) IS still in scope — it cleans up the photo before
  sculptok sees it. Post-sculptok shaping is NOT in scope.
