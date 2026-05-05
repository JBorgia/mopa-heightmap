import numpy as np

from zoedepth.laser.detail import (
    DetailSettings,
    apply_detail_injection,
    settings_from_mapping,
)


def _flat_depth(value=0.5, shape=(32, 32)):
    return np.full(shape, value, dtype=np.float32)


def _checker_photo(shape=(32, 32)):
    """Photo with strong local contrast — luminance varies a lot."""
    rng = np.random.default_rng(7)
    rgb = (rng.uniform(0.0, 1.0, size=(*shape, 3)) * 255).astype(np.uint8)
    return rgb


def test_detail_off_is_passthrough():
    h = _flat_depth()
    rgb = _checker_photo()
    out = apply_detail_injection(h, rgb, DetailSettings(mode="off", strength=0.5))
    np.testing.assert_array_equal(out, h)


def test_detail_strength_zero_is_passthrough():
    h = _flat_depth()
    rgb = _checker_photo()
    out = apply_detail_injection(h, rgb, DetailSettings(mode="luminance", strength=0.0))
    np.testing.assert_array_equal(out, h)


def test_detail_luminance_modifies_pixels():
    h = _flat_depth(value=0.5)
    rgb = _checker_photo()
    out = apply_detail_injection(
        h, rgb, DetailSettings(mode="luminance", strength=0.5, subject_mask=False)
    )
    # Output must differ from the flat input but stay in [0,1].
    assert not np.allclose(out, h)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_detail_highpass_preserves_mean_approximately():
    """High-pass adds zero-mean detail; the mean should barely shift."""
    h = _flat_depth(value=0.6)
    rgb = _checker_photo()
    out = apply_detail_injection(
        h, rgb,
        DetailSettings(mode="highpass", strength=0.5, highpass_radius=5, subject_mask=False),
    )
    assert not np.allclose(out, h)
    assert abs(out.mean() - h.mean()) < 0.05


def test_detail_subject_mask_protects_background():
    """With subject_mask on, far/deep pixels (heightmap=0) should be untouched."""
    h = np.zeros((32, 32), dtype=np.float32)  # entirely background
    h[8:24, 8:24] = 0.8                       # subject patch
    rgb = _checker_photo()
    out = apply_detail_injection(
        h, rgb,
        DetailSettings(mode="luminance", strength=0.7, subject_mask=True),
    )
    # Background (all-zero region outside the patch) must remain ~0.
    bg_mask = np.ones_like(h, dtype=bool)
    bg_mask[8:24, 8:24] = False
    np.testing.assert_allclose(out[bg_mask], 0.0, atol=1e-3)
    # Subject region is modified.
    assert not np.allclose(out[8:24, 8:24], 0.8)


def test_detail_resizes_photo_to_heightmap_shape():
    h = _flat_depth(value=0.5, shape=(40, 30))
    rgb = (np.random.default_rng(0).uniform(0, 255, (160, 120, 3))).astype(np.uint8)
    out = apply_detail_injection(
        h, rgb, DetailSettings(mode="luminance", strength=0.4, subject_mask=False)
    )
    assert out.shape == h.shape


def test_settings_from_mapping_parses_keys():
    cfg = settings_from_mapping({
        "detail_mode": "BOTH",
        "detail_strength": 0.6,
        "detail_highpass_radius": 12,
        "detail_subject_mask": False,
        "detail_invert": True,
    })
    assert cfg.mode == "both"
    assert cfg.strength == 0.6
    assert cfg.highpass_radius == 12
    assert cfg.subject_mask is False
    assert cfg.invert is True


def test_settings_from_mapping_rejects_unknown_mode():
    cfg = settings_from_mapping({"detail_mode": "bogus", "detail_strength": 0.4})
    # Falls back to default "off" rather than raising.
    assert cfg.mode == "off"
    assert cfg.strength == 0.4
