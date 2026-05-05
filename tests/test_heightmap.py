import numpy as np
import pytest

from zoedepth.laser.heightmap import (
    apply_tone_curve,
    floyd_steinberg_dither,
    joint_bilateral_refine,
    normalize_depth,
    orient_for_lightburn,
    posterize_for_passes,
    process_depth_to_heightmap,
    to_uint8,
    to_uint16,
)


@pytest.fixture
def synthetic_depth():
    # Gradient depth: 0 (near) -> 10 (far)
    grad = np.linspace(0.0, 10.0, 64, dtype=np.float32)
    return np.tile(grad, (32, 1))


def test_normalize_clips_to_unit_range(synthetic_depth):
    out = normalize_depth(synthetic_depth, near_percentile=5.0, far_percentile=95.0)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Strict monotonic across the gradient axis.
    row = out[0]
    assert np.all(np.diff(row) >= -1e-6)


def test_orient_black_is_deep_inverts():
    row = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    near_zero_far_one = np.tile(row, (4, 1))
    flipped = orient_for_lightburn(near_zero_far_one, black_is_deep=True)
    # Near (smaller depth = closer = high relief = bright).
    assert flipped[0, 0] > flipped[0, -1]


def test_apply_tone_curve_respects_limits():
    row = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    x = np.tile(row, (4, 1))
    y = apply_tone_curve(x, gamma=1.0, contrast=1.0, midtone_boost=0.0,
                         deep_limit=0.1, surface_limit=0.9)
    assert y.min() >= 0.1 - 1e-6
    assert y.max() <= 0.9 + 1e-6


def test_process_pipeline_produces_unit_float(synthetic_depth):
    settings = {
        "near_percentile": 5.0, "far_percentile": 95.0,
        "gamma": 0.8, "contrast": 1.0, "midtone_boost": 0.0,
        "deep_limit": 0.04, "surface_limit": 0.96,
        "black_is_deep": True,
        "flatten_background": False,
        "background_threshold": 0.9, "background_value": 1.0,
        "smooth": "off", "smooth_diameter": 5, "smooth_strength": 0.05,
        "sharpen": 0.0, "sharpen_sigma": 1.0,
    }
    h = process_depth_to_heightmap(synthetic_depth, settings)
    assert h.dtype == np.float32
    assert h.shape == synthetic_depth.shape
    assert 0.0 <= h.min() and h.max() <= 1.0


def test_quantization_round_trips():
    x = np.linspace(0.0, 1.0, 128, dtype=np.float32).reshape(8, 16)
    u8 = to_uint8(x)
    u16 = to_uint16(x)
    assert u8.dtype == np.uint8
    assert u16.dtype == np.uint16
    assert u8.max() == 255 and u16.max() == 65535
    assert u8.min() == 0 and u16.min() == 0

# Phase 2 additions


def test_floyd_steinberg_dither_produces_quantized_levels():
    x = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (16, 1))
    out = floyd_steinberg_dither(x, levels=4)
    assert out.dtype == np.float32
    unique_vals = np.unique(out)
    # 4 levels means values from {0, 1/3, 2/3, 1}.
    assert unique_vals.size <= 4
    for v in unique_vals:
        assert any(abs(v - target) < 1e-6 for target in (0.0, 1 / 3, 2 / 3, 1.0))


def test_floyd_steinberg_preserves_mean_brightness():
    x = np.full((24, 24), 0.5, dtype=np.float32)
    out = floyd_steinberg_dither(x, levels=2)
    # With 2 levels {0,1}, mean should still hover near 0.5.
    assert abs(out.mean() - 0.5) < 0.05


def test_joint_bilateral_refine_no_guide_is_no_op():
    x = np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (8, 1))
    out = joint_bilateral_refine(x, None)
    np.testing.assert_array_equal(out, x.astype(np.float32))


def test_joint_bilateral_refine_runs_with_guide():
    x = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (16, 1))
    guide = np.zeros((16, 32, 3), dtype=np.uint8)
    guide[:, 16:] = 200
    out = joint_bilateral_refine(x, guide, diameter=5, sigma_color=0.1, sigma_space=3.0)
    assert out.dtype == np.float32
    assert out.shape == x.shape
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_posterize_for_passes_quantizes_to_n_plus_one_levels():
    x = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (4, 1))
    out = posterize_for_passes(x, n_passes=4)
    unique = np.unique(out)
    # n_passes=4 → 5 distinct depths including 0.
    assert len(unique) == 5
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_posterize_preserves_monotonicity():
    x = np.tile(np.linspace(0.0, 1.0, 100, dtype=np.float32), (1, 1))
    out = posterize_for_passes(x, n_passes=8)
    diffs = np.diff(out[0])
    assert (diffs >= -1e-6).all()


def test_posterize_one_pass_is_binary():
    x = np.tile(np.linspace(0.0, 1.0, 50, dtype=np.float32), (2, 1))
    out = posterize_for_passes(x, n_passes=1)
    assert set(np.unique(out).tolist()).issubset({0.0, 1.0})


def test_process_pipeline_honors_posterize(synthetic_depth):
    settings = {"black_is_deep": True, "posterize_passes": 3}
    out = process_depth_to_heightmap(synthetic_depth, settings)
    # 3 passes → at most 4 distinct levels.
    assert len(np.unique(out)) <= 4

