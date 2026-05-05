---
name: photo-luminance highpass is the right detail source for portrait photos
description: For laser bas-relief from photo portraits, photo-luminance highpass at low strength + small radius beats the geometric FC-normals composite, contradicting an earlier IMPLEMENTATION_PLAN claim.
type: feedback
---

For laser bas-relief from **photo** portraits (i.e., real photographs, not orthographic 3D renders), prefer photo-luminance high-pass injection over the FC-integrated-normals composite for surface detail:

- `detail_mode=highpass`, `detail_strength=0.10–0.20`, `detail_highpass_radius=5–7`.
- Disable `relief_enabled` (the normals→FC composite).

**Why:** The IMPLEMENTATION_PLAN.md §2 ("Why we are rebuilding") flagged photo-luminance injection as "non-physical — produces ink-blot artefacts" and dropped α to 0.05. That was right at α=0.4 with luminance-blend mode, but at α≈0.10–0.20 in **highpass** mode with a small radius, the high-pass captures only feature edges (eyes, beard, embroidery, fabric stitches) without the ink-blot effect. Confirmed by visual comparison against sculptok.png on the king reference photo.

The geometric alternative (DSINE normals → Frankot-Chellappa integration → composite) doesn't work for portrait photos because DSINE caps at 480 px inference resolution. By the time it's resampled to a 1024-pixel render, face features occupy fewer than 100 pixels of normal-map output — there's no high-frequency content to extract. The photo, by contrast, carries detail at native resolution (often 4000+ pixels wide).

**How to apply:** When tuning portrait output, treat photo-luminance highpass as the primary surface-detail knob, not the safety-off fallback. The `relief_*` setting stays available for non-photo subjects (3D renders, line-art-derived heightmaps) where the depth backbone produces only smooth form.
