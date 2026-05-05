---
name: sculptok parity ceiling — what we tried and what works
description: After exhausting open-source approaches (Real-ESRGAN, DAv2_Large, ControlNet+SD1.5+bas-relief, TripoSR mesh) we cannot match sculptok's raw visual detail at 4 GB VRAM. Don't repeat any of these experiments without a real bas-relief LoRA or higher-VRAM GPU.
type: feedback
---

For portrait bas-relief on the king reference image (`assets/360_F_320748738_*.jpg`, 539×360) the achievable quality ceiling on **open models + 4 GB VRAM (Quadro P2000)** is "very close to sculptok on macro form, a notch softer in fur/beard/face micro-detail." Stop trying to match sculptok pixel-for-pixel in this configuration.

Things that **work** and should stay in `sculptok_portrait` (validated May 2026):

1. **Real-ESRGAN x4 super-resolution before depth** — biggest single contributor; closed ~60 % of the gap.
2. **DAv2_Base + BiRefNet mask + photo-luminance highpass at 0.20–0.22 + Kerber unsharp + face_relief**.
3. **Tighter BiRefNet feather (1 px) + harder sharpen (0.6)** — kills the silhouette halo, lifts edges.

Things that **do not move the needle** and should NOT be re-attempted without new evidence:

4. **DAv2_Base → DAv2_Large** — same depth output. Don't waste the NC opt-in for this image class.
5. **ControlNet-Depth + SDXL + bas-relief prompt** — too heavy for 4 GB VRAM (forces CPU, ~30 min/render).
6. **ControlNet-Depth + SD1.5 (Dreamshaper-8) + bas-relief prompt** — fits in 4 GB, ~10 min/render. Output is essentially identical to baseline. Generic prompt-only stylization without a fine-tuned bas-relief LoRA doesn't generate sculptural surface texture; DAv2 then re-extracts a depth map that looks like the original. **Code stays in `relief_stylizer.py` as opt-in.**
7. **TripoSR mesh-based depth (image → 3-D mesh → orthographic Z)** — `zoedepth/laser/mesh_depth.py` plus a PyMCubes shim. **Mesh quality is too poor for complex portrait poses (raised-arms V-pose).** The 44k-triangle mesh is a blobby approximation; orthographic projection gives a vague humanoid silhouette, not sculptural detail. The vendored architecture (`vendor/triposr/`) was **removed during cleanup** — re-clone instructions live at the top of `mesh_depth.py`. The wrapper itself stays as opt-in scaffolding so revisiting mesh-based depth (Hunyuan3D, TRELLIS, or a better TripoSR variant) when we have ≥ 8 GB VRAM is a swap-the-loader change rather than starting over.

**Why we lose the last 15-20 %:** sculptok almost certainly uses one of:
- A bas-relief-fine-tuned diffusion LoRA (custom-trained on relief reference pairs).
- A higher-resource image-to-3D model (Hunyuan3D-2, TRELLIS) that needs 8-16 GB VRAM.
- Manual sculptural cleanup by a human operator.

We have access to none of these with the current hardware envelope.

**Where we win even with the gap:** every engraving-side dimension sculptok doesn't even attempt — physical calibration LUT auto-fit, multi-pass `.lbrn2`/`.clb` export, burn-time estimation, face-region sculptural weighting (`face_relief`), QA failure detection, signature pass, profile system, per-material cut-setting libraries. **This is the real shipping value.** Use sculptok for marketing-quality king-portrait demos; use mopa-heightmap for the actual production tooling.

**How to apply this memory:**

* If a future session aims to match sculptok pixel-for-pixel on this image class with the current hardware: **don't repeat steps 4, 5, or 6.** None of them moved the needle.
* If the user gets a bas-relief LoRA from Civitai or HF, plug it into `relief_stylize_backend` and try again — that's the one path we expect to actually close the gap.
* If the user upgrades to ≥ 8 GB VRAM, retry with Hunyuan3D-2 instead of TripoSR — the mesh quality should be substantially better.

**Stable shipping state:** `sculptok_portrait` profile uses Real-ESRGAN + DAv2_Base + tight feather + 0.6 sharpen + face_relief + Kerber unsharp. ~60-90 s render time. 80-85 % of sculptok visual quality plus the strongest engraving-side toolchain available. That's the right tradeoff.
