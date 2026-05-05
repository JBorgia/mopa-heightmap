---
name: research findings — meshy.ai and SOTA bas-relief landscape
description: Two synthesis reports from research agents (May 2026) that informed the face_relief + depth_unsharp + Sapiens additions. Useful for future "what should we add next?" questions.
type: reference
---

Two research agents surveyed the landscape on 2026-05-04. Key findings:

## Meshy.ai
Closed product, generic 3D-diffusion architecture, aimed at full closed meshes (NOT 2.5D bas-relief). Don't reverse-engineer it. Borrowable ideas: multi-view synthesis as a preprocessor, explicit delighting / albedo separation, curvature + AO as auxiliary maps, learned final "healing" pass, automatic frontalization. Their displacement maps are coarser than our photo-luminance high-pass.

## State of the art for image-to-bas-relief (priority-ordered for laser engraving)

**The moat — face-aware per-region depth weighting** (built as `zoedepth/laser/face_relief.py`). No open tool does this end-to-end. It's the single biggest visual differentiator vs. Sculptok / VistaSculpt / ReliefMod / Carveco / Cura / GIMP — they don't have a sculptor's hand.

**Other high-impact additions implemented:**
- `Sapiens_Depth_1B` (Meta, ECCV 2024, CC-BY-NC) — portrait SOTA at 1024 native, +22.4% RMSE on Hi4D. Registered in `backends.py`.
- `Sapiens_Normal_1B` — replaces the DSINE 480 px ceiling for FC-integrated relief. Registered in `normals.py`.
- Kerber gradient-domain compression — `depth_unsharp.py`. The "shallow but sharp" recipe.

**Considered and skipped:**
- Hunyuan3D-2 / TripoSR / Stable Fast 3D / TRELLIS-2 / InstantMesh — all have triplane bottlenecks that destroy sub-mm detail. Direct depth at native resolution beats orthographic-projection-from-mesh for our use case.
- Marigold-IID-Appearance delighting — high-impact for jewelry/glossy materials but not yet wired (4 hours of work). Add when we get reports of specular-as-pit artifacts.
- Multi-view fusion (Zero123++/SV3D) — heavy (~30-60s/portrait, 16 GB VRAM) for marginal portrait improvement.

**Failure modes to watch for:**
1. Background floater behind head (mitigated by BiRefNet hard-flatten).
2. Hollow cheeks (DA-V2 misreads soft shadow as concavity) — Sapiens-Depth solves this.
3. Specular-as-pit on jewelry/eyes/glossy lips — needs Marigold-IID delighting.
4. Floating hair (DC offset error) — needs hair-region DC clamp.
5. Mirror-flipped depth (rare in Marigold) — flag if L/R asymmetric without lighting cause.
6. Earring / jewelry as depth pits — specular detection + inpaint.

The full agent reports are in the conversation history of the May 2026 session that built these features.
