import numpy as np

from zoedepth.laser.autofit import autofit_overrides_from_depth


def _make_subject_plus_background(subject_depth=1.5, bg_depth=4.5, bg_frac=0.6, shape=(64, 64)):
    rng = np.random.default_rng(0)
    n_pixels = shape[0] * shape[1]
    n_bg = int(n_pixels * bg_frac)
    n_subject = n_pixels - n_bg
    subject = rng.normal(loc=subject_depth, scale=0.2, size=n_subject)
    background = rng.normal(loc=bg_depth, scale=0.05, size=n_bg)
    depth = np.concatenate([subject, background])
    rng.shuffle(depth)
    return depth.reshape(shape).astype(np.float32)


def test_autofit_with_dominant_far_background_clips_aggressively():
    depth = _make_subject_plus_background(bg_frac=0.6)
    out = autofit_overrides_from_depth(depth)

    assert "near_percentile" in out
    assert "far_percentile" in out
    # Strong background should pull far_pct WELL below 95.
    assert out["far_percentile"] < 80.0
    # Subject mass starts close to the bottom; near_pct should be small.
    assert out["near_percentile"] <= 10.0
    # Reasonable gamma / reserves.
    assert 0.3 <= out["gamma"] <= 2.0
    assert 0.0 <= out["deep_limit"] < out["surface_limit"] <= 1.0


def test_autofit_with_no_background_uses_wide_range():
    rng = np.random.default_rng(1)
    depth = rng.uniform(1.0, 3.0, size=(64, 64)).astype(np.float32)
    out = autofit_overrides_from_depth(depth)
    # No far peak -> roughly the default 5/95 percentile shape.
    assert out["far_percentile"] >= 95.0 - 5.0  # allow a little wiggle
    assert out["near_percentile"] <= 10.0


def test_autofit_handles_degenerate_input():
    flat = np.full((32, 32), 2.0, dtype=np.float32)
    assert autofit_overrides_from_depth(flat) == {}
    empty = np.array([], dtype=np.float32).reshape(0, 0)
    assert autofit_overrides_from_depth(empty) == {}


def test_autofit_includes_detail_injection_defaults():
    depth = _make_subject_plus_background()
    out = autofit_overrides_from_depth(depth)
    assert out["detail_mode"] in ("luminance", "highpass", "both")
    assert 0.0 < out["detail_strength"] <= 1.0
    assert out["detail_subject_mask"] is True
