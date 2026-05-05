---
name: face-aware per-region depth weighting is the moat
description: For portrait laser engraving, hand-tuned per-region depth offsets (sculptor-style nose lift, nostril deepening, eye-socket recess) are MOPA-Heightmap's key visual differentiator vs every other open or commercial tool.
type: feedback
---

For portrait bas-relief, **face-aware per-region depth weighting** (`zoedepth/laser/face_relief.py`) is what makes our output recognisably better than Sculptok / VistaSculpt / ReliefMod / Carveco / Cura / GIMP. None of those do it.

The recipe is a sculptor's choices:
- Deepen nostrils (-0.18) and eye sockets (-0.10).
- Raise nose tip (+0.07), bridge (+0.05), cheekbones (+0.04).
- Slight chin underside recess (-0.04).
- Subtle lip and brow lifts.

Driven by MediaPipe FaceMesh's 478 landmarks, applied as Gaussian splats. No-op when no face is detected, so non-portrait subjects pass through untouched.

**Why:** Confirmed by both research agents (2026-05-04) that this is the single biggest visual lever we're missing vs published SOTA, and that no open tool implements it end-to-end. Validated on the king reference photo (`assets/360_F_320748738_*.jpg`) — at `strength=1.0` with photo highpass at 0.22, the resulting heightmap matches sculptok.png's level of facial sculpting.

**How to apply:** Always recommend `face_relief_enabled: true` in the `sculptok_portrait` profile for human-subject jobs. For non-face subjects (products, animals, abstract art) it auto-no-ops, so leaving it on by default is safe. If a user wants more aggressive sculpting, push `face_relief_strength` to 1.3-1.5 (capped at 2.0).
