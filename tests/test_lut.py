"""Tests for the calibration LUT."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.lut import CalibrationLUT, lut_from_profile


def _identity_samples(max_depth_um: float = 100.0):
    grays = [0, 25, 51, 76, 102, 128, 153, 178, 204, 229, 255]
    depths = [g / 255.0 * max_depth_um for g in grays]
    return list(zip(grays, depths))


def test_from_measurements_default_gray_levels():
    depths = [0.0, 5.0, 10.0, 18.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0]
    lut = CalibrationLUT.from_measurements(depths, note="test")
    assert len(lut.samples) == 11
    assert lut.note == "test"
    assert lut.max_depth_um == 120.0


def test_from_measurements_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        CalibrationLUT.from_measurements([0, 10], gray_levels=[0, 50, 100])


def test_from_measurements_too_few_points_raises():
    with pytest.raises(ValueError):
        CalibrationLUT.from_measurements([0.0])


def test_dict_round_trip():
    lut = CalibrationLUT.from_measurements([0.0, 10.0, 25.0], gray_levels=[0, 128, 255])
    data = lut.to_dict()
    lut2 = CalibrationLUT.from_dict(data)
    np.testing.assert_array_equal(lut._arrays()[0], lut2._arrays()[0])
    np.testing.assert_array_equal(lut._arrays()[1], lut2._arrays()[1])


def test_from_dict_accepts_dict_entries():
    payload = {
        "samples": [
            {"gray": 0, "depth_um": 0},
            {"gray": 128, "depth_um": 50},
            {"gray": 255, "depth_um": 100},
        ]
    }
    lut = CalibrationLUT.from_dict(payload)
    assert lut.max_depth_um == 100.0


def test_gray_to_depth_interpolates():
    lut = CalibrationLUT(samples=[(0, 0.0), (128, 50.0), (255, 100.0)])
    # Halfway between 128 and 255 should give ~75 µm.
    assert abs(float(lut.gray_to_depth_um(192)) - 75.0) < 1.0


def test_apply_identity_lut_is_near_no_op():
    lut = CalibrationLUT(samples=_identity_samples(100.0))
    x = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (16, 1))
    out = lut.apply(x, target_depth_um=100.0)
    assert out.shape == x.shape
    # Identity LUT should produce a near-identity remap.
    assert np.max(np.abs(out - x)) < 0.01


def test_apply_compresses_under_nonlinear_lut():
    # Heavily nonlinear: most depth happens at high gray levels.
    samples = [(0, 0.0), (128, 5.0), (255, 100.0)]
    lut = CalibrationLUT(samples=samples)
    x = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (16, 1))
    out = lut.apply(x, target_depth_um=100.0)
    # Output must remain in [0,1] and differ from input.
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, x, atol=0.01)


def test_apply_no_target_uses_max_depth():
    lut = CalibrationLUT(samples=[(0, 0.0), (255, 80.0)], max_depth_um=80.0)
    x = np.full((4, 4), 0.5, dtype=np.float32)
    out = lut.apply(x)  # target_depth_um defaults to max_depth_um
    assert out.shape == x.shape


def test_apply_zero_target_returns_input():
    lut = CalibrationLUT(samples=[(0, 0.0), (255, 80.0)])
    x = np.full((4, 4), 0.5, dtype=np.float32)
    out = lut.apply(x, target_depth_um=0.0)
    np.testing.assert_array_equal(out, x)


def test_apply_requires_2d():
    lut = CalibrationLUT(samples=[(0, 0.0), (255, 100.0)])
    with pytest.raises(ValueError):
        lut.apply(np.array([0.5, 0.5], dtype=np.float32))


def test_lut_from_profile_returns_none_when_absent():
    assert lut_from_profile({"name": "x"}) is None


def test_lut_from_profile_parses_block():
    profile = {
        "calibration_lut": {
            "note": "n",
            "samples": [[0, 0], [128, 50], [255, 100]],
        }
    }
    lut = lut_from_profile(profile)
    assert isinstance(lut, CalibrationLUT)
    assert lut.note == "n"


def test_lut_monotonic_clamping_in_arrays():
    # Out-of-order or non-monotonic input depth values get accumulated upward.
    lut = CalibrationLUT(samples=[(0, 0), (128, 50), (255, 30)])  # 30 < 50
    g, d = lut._arrays()
    assert list(d) == [0.0, 50.0, 50.0]
