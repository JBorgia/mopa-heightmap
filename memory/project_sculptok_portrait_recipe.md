---
name: sculptok-portrait recipe
description: The validated recipe that gets MOPA-Heightmap to sculptok-grade portrait output, and what actually moves the needle vs what was theory.
type: project
---

The recipe that reaches sculptok-grade output on portrait photos:

1. **Subject mask**: BiRefNet (default-on, MIT, ~3.5 GB VRAM) for the cleanest cutout, or rembg (CPU) as fallback.
2. **Depth backbone**: `DAv2_Base` at `inference_resolution=1024`. ZoeD_NK is too low-resolution (384 px native) to resolve facial features.
3. **Photo-luminance high-pass**: `detail_mode=highpass`, `detail_strength=0.20`, `detail_highpass_radius=5`. This is what surfaces face, beard, embroidery, buttons, fabric folds.
4. **FC-integrated-normals composite (`relief_*`)**: OFF for photos. Current DSINE caps at 480 px inference, so it has no usable high-frequency content to add. Keep available and well-tuned for non-photo subjects (orthographic 3D renders, line-art-derived heightmaps) where the depth lacks micro-texture.

Profile: `profiles/sculptok_portrait.yaml`. Smoke bench: `smoke_sculptok.py`.

**Why:** Validated by side-by-side comparison against `assets/sculptok.png` on the king reference photo (`assets/360_F_320748738_*.jpg`) on 2026-05-04. All sculptok features reproduce except fur trim texture, which is slightly softer than sculptok — closing that gap needs DAv2_Large (CC-BY-NC) or a higher-resolution normal estimator than DSINE.

**How to apply:** When the user wants portrait/sculpture/medallion bas-relief output, point them at `--profile sculptok_portrait --model DAv2_Base`. If they ask why relief is off in the profile, the reason is normal-estimator resolution, not architecture; and the photo-luminance high-pass is doing the surface-detail work for them.
