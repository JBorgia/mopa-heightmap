"""Regression tests for the pre-sculptok prep wiring.

Before this change the wizard's "Pre-sculptok prep" toggles only ran
during ``/render`` (i.e. AFTER sculptok had already produced the heightmap),
so they couldn't influence depth-map quality. This module pins:

  * ``HeightmapService.prepare_input()`` is callable independently of
    ``render()`` and produces a conditioned image + optional subject mask.
  * Solid background patterns scrub the background to a flat colour
    (rather than blending with the photo's mean grey, which is what the
    pattern-based composite does).
  * The wizard's existing input toggles flow through ``prepare_input``
    untouched (so /render parity is preserved).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from mopa.service import HeightmapService


def _photo(w: int = 64, h: int = 64) -> Image.Image:
    """Synthetic photo with a clear "subject" in the centre and a busy
    grey background — gives the conditioning code something to bite on.

    Uses a gradient inside the subject so the threshold mask backend's
    90th-percentile cutoff catches a meaningful slice (a flat-colour
    subject would either be 100% included or 100% excluded).
    """
    arr = np.full((h, w, 3), 50, dtype=np.uint8)               # dark grey background
    # Bright subject with a small luma gradient
    yy = np.arange(h // 4, 3 * h // 4)[:, None]
    xx = np.arange(w // 4, 3 * w // 4)[None, :]
    sub = 200 + ((yy - h // 2) ** 2 + (xx - w // 2) ** 2 < (w // 4) ** 2).astype(np.uint8) * 50
    sub = np.clip(sub, 0, 255).astype(np.uint8)
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = sub[..., None]
    return Image.fromarray(arr, "RGB")


def test_prepare_input_returns_conditioned_image_and_no_mask_by_default():
    svc = HeightmapService()
    img = _photo()
    out, alpha, image_hash = svc.prepare_input(img, {})
    assert isinstance(out, Image.Image)
    assert out.size == img.size  # no auto-crop without explicit aspect
    assert alpha is None
    assert isinstance(image_hash, str) and len(image_hash) > 0


def test_prepare_input_runs_clahe_when_enabled_changes_pixel_values():
    """CLAHE should produce a visibly different image — pin that the
    setting is actually wired through, not silently dropped."""
    svc = HeightmapService()
    img = _photo()
    plain, _alpha, _h = svc.prepare_input(img, {})
    boosted, _alpha, _h = svc.prepare_input(img, {"input_clahe": True})
    plain_arr = np.asarray(plain.convert("L"))
    boosted_arr = np.asarray(boosted.convert("L"))
    # CLAHE on a flat-grey background bumps local contrast — the mean
    # difference is small but at least one pixel must differ.
    assert not np.array_equal(plain_arr, boosted_arr)


def test_composite_background_solid_black_uses_full_intensity():
    """``_composite_background`` is the engine the bg-replace UX rides on.
    Test it directly with a known mask so we don't depend on the masker
    backends agreeing about what counts as subject."""
    svc = HeightmapService()
    photo = _photo(w=64, h=64)
    # Hand-crafted alpha: subject = inner 32×32, background = everything else.
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    out = svc._composite_background(photo, alpha, {
        "background_pattern": "solid_black",
    })
    out_arr = np.asarray(out.convert("L"))
    # Corner = pure background, must be black (solid_* forces intensity=1).
    assert out_arr[0, 0] == 0
    assert out_arr[-1, -1] == 0
    # Centre = subject, must keep the original photo content.
    assert out_arr[32, 32] > 150


def test_composite_background_solid_white_uses_full_intensity():
    svc = HeightmapService()
    photo = _photo(w=64, h=64)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    out = svc._composite_background(photo, alpha, {
        "background_pattern": "solid_white",
    })
    out_arr = np.asarray(out.convert("L"))
    assert out_arr[0, 0] == 255
    assert out_arr[-1, -1] == 255


def test_composite_background_procedural_pattern_blends_with_photo_mean():
    """Decorative procedural patterns (guilloche/stripes/etc.) keep the
    soft blend so they read as texture, not a hard cut. Pin that the
    solid-* fast path doesn't apply to them."""
    svc = HeightmapService()
    photo = _photo(w=64, h=64)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    out = svc._composite_background(photo, alpha, {
        "background_pattern": "stripes",
        "background_intensity": 0.5,
    })
    out_arr = np.asarray(out.convert("L"))
    # Background should NOT be 0 or 255 — it's a blend.
    assert 20 < out_arr[0, 0] < 240


def test_prepare_input_unknown_pattern_silently_passes_through():
    """An unknown pattern name shouldn't crash; the bg-composite step
    just no-ops and returns the conditioned image as-is."""
    svc = HeightmapService()
    img = _photo()
    out, _alpha, _hash = svc.prepare_input(img, {
        "subject_mask_enabled": True,
        "subject_mask_backend": "threshold",
        "background_pattern": "not_a_real_pattern",
    })
    plain, _alpha, _h = svc.prepare_input(img, {
        "subject_mask_enabled": True,
        "subject_mask_backend": "threshold",
    })
    assert np.array_equal(np.asarray(out), np.asarray(plain))


def test_render_still_works_after_prepare_input_refactor(tmp_path):
    """End-to-end: HeightmapService.render() must keep working — the
    prepare_input helper was extracted FROM render(), so a regression
    here would mean both the new sculptok-prep flow AND the existing
    render flow are broken."""
    svc = HeightmapService()
    img = _photo()
    # Drop a fake heightmap file in the temp dir
    hm_path = tmp_path / "hm.png"
    hm = (np.linspace(0.2, 0.9, 64 * 64).reshape(64, 64) * 65535).astype(np.uint16)
    Image.fromarray(hm, mode="I;16").save(hm_path)
    result = svc.render(img, {
        "external_heightmap_path": str(hm_path),
        "input_clahe": True,
    })
    assert result.heightmap.shape == (64, 64)
    assert result.preview_image is not None
    assert result.conditioned is not None
