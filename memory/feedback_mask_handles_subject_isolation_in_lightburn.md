---
name: subject isolation lives in the LightBurn mask, not the heightmap
description: Sculptok output fills the frame by design. The subject mask is a separate deliverable that LightBurn applies at engrave time. Don't write QA checks or pipeline logic that assume the heightmap has a flat background plane.
type: feedback
---

The new sculptok-only workflow does NOT flatten the background in the
heightmap itself. Sculptok produces a full-frame relief and we ship it
unchanged. Subject/background separation is handled at engrave time by:
- a separate ``mask.png`` deliverable in the bundle (alpha PNG), and
- LightBurn's per-layer cut settings (the user disables passes outside
  the subject silhouette).

**Why:** the user pointed out that my smoke-test QA finding
"Subject covers 100.0% of the frame; no background plane" was a false
positive — full coverage is correct, the mask file is what carries
the subject silhouette, and LightBurn applies it. Encoded as the
"separate mask deliverable, not heightmap modifier" architecture
(see commits c2f2810, f328f53).

**How to apply:**
- QA checks must NOT assume the heightmap has bg pixels at
  ``background_value``. Most depth-era checks (``subject_fills_frame``,
  ``subject_too_small``, ``bg_floater``) operate on that assumption
  and will misfire on sculptok output.
- When adding a subject-coverage QA check, run it on the
  ``subject_alpha`` mask, not on the heightmap.
- Don't add code that mutates the heightmap to flatten a background.
  Sculptok already produced the right thing; touching it re-introduces
  the original bug ("you destroyed the black background").
- The mask is a *deliverable*, not a modifier. Treat it the same way
  you'd treat a vector cut-line: a sibling PNG in ``final/`` that the
  user references in their LightBurn project.
