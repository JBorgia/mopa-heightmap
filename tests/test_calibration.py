"""Tests for :mod:`mopa.calibration`."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mopa.calibration import (
    DEFAULT_RAMP_AXIS,
    DEFAULT_RAMP_GRAY_LEVELS,
    RampMeasurement,
    calibration_lut_from_ramp_photo,
    measure_engraved_ramp,
)
from mopa.lut import CalibrationLUT


def _synthetic_engraved_ramp(
    n_steps: int = 11, width: int = 880, height: int = 60, axis: str = "horizontal",
    invert: bool = False,
) -> Image.Image:
    """Generate a fake photo of an engraved ramp with a clean monotonic gradient."""
    bands = np.linspace(0.95, 0.05, n_steps) if not invert else np.linspace(0.05, 0.95, n_steps)
    if axis == "horizontal":
        canvas = np.zeros((height, width), dtype=np.float32)
        band_w = width // n_steps
        for i, level in enumerate(bands):
            start = i * band_w
            end = width if i == n_steps - 1 else (i + 1) * band_w
            canvas[:, start:end] = level
    else:
        canvas = np.zeros((width, height), dtype=np.float32)
        band_w = width // n_steps
        for i, level in enumerate(bands):
            start = i * band_w
            end = width if i == n_steps - 1 else (i + 1) * band_w
            canvas[start:end, :] = level
    return Image.fromarray((canvas * 255).astype(np.uint8), "L").convert("RGB")


# ----------------------------------------------------------- constants

def test_default_gray_levels_match_ramp():
    assert DEFAULT_RAMP_GRAY_LEVELS == (255, 230, 204, 179, 153, 128, 102, 77, 51, 26, 0)
    assert DEFAULT_RAMP_AXIS == "horizontal"


# ----------------------------------------------------------- measure

def test_measure_engraved_ramp_produces_monotonic_depths():
    photo = _synthetic_engraved_ramp()
    m = measure_engraved_ramp(photo, max_depth_um=120.0)
    assert isinstance(m, RampMeasurement)
    assert len(m.band_depth_um) == 11
    # Monotonic non-decreasing.
    deltas = np.diff(m.band_depth_um)
    assert (deltas >= -1e-3).all(), f"non-monotonic: {m.band_depth_um}"
    # Endpoints scale to max_depth_um.
    assert m.band_depth_um[0] < 5.0
    assert m.band_depth_um[-1] > 100.0


def test_measure_engraved_ramp_normalised_when_no_max_depth():
    photo = _synthetic_engraved_ramp()
    m = measure_engraved_ramp(photo)  # no max_depth_um
    assert min(m.band_depth_um) >= 0.0
    assert max(m.band_depth_um) <= 1.0 + 1e-3


def test_measure_engraved_ramp_invert_flag():
    photo = _synthetic_engraved_ramp(invert=True)
    m = measure_engraved_ramp(photo, max_depth_um=120.0, invert=True)
    # First band still surface (depth 0), last still deepest.
    assert m.band_depth_um[0] < 5.0
    assert m.band_depth_um[-1] > 100.0


def test_measure_engraved_ramp_vertical_axis():
    photo = _synthetic_engraved_ramp(axis="vertical")
    m = measure_engraved_ramp(photo, axis="vertical", max_depth_um=80.0)
    assert len(m.band_depth_um) == 11


def test_measure_rejects_indistinguishable_bands():
    photo = Image.new("RGB", (220, 60), (128, 128, 128))
    with pytest.raises(ValueError, match="indistinguishable"):
        measure_engraved_ramp(photo)


def test_measure_rejects_n_steps_below_two():
    photo = _synthetic_engraved_ramp()
    with pytest.raises(ValueError, match="n_steps"):
        measure_engraved_ramp(photo, n_steps=1)


def test_measure_with_crop():
    """Crop to the first 5 bands and verify only those are sampled."""
    photo = _synthetic_engraved_ramp(width=880)
    crop = (0, 0, 400, 60)
    m = measure_engraved_ramp(photo, n_steps=5, crop=crop, max_depth_um=120.0)
    assert len(m.band_depth_um) == 5


# ----------------------------------------------------------- LUT factory

def test_calibration_lut_from_ramp_photo_returns_lut_and_measurement():
    photo = _synthetic_engraved_ramp()
    lut, measurement = calibration_lut_from_ramp_photo(photo, max_depth_um=120.0)
    assert isinstance(lut, CalibrationLUT)
    assert isinstance(measurement, RampMeasurement)
    assert lut.max_depth_um == pytest.approx(120.0, abs=2.0)
    assert len(lut.samples) == 11


def test_calibration_lut_inverts_correctly_at_endpoints():
    photo = _synthetic_engraved_ramp()
    lut, _ = calibration_lut_from_ramp_photo(photo, max_depth_um=120.0)
    # Engraver-frame convention: gray=0 → no engraving (depth=0),
    # gray=255 → full engraving (depth ≈ max_depth_um). This pairs with
    # apply()'s (1 - v) inversion on a black_is_deep heightmap.
    assert lut.gray_to_depth_um(0.0) < 5.0
    assert lut.gray_to_depth_um(255.0) > 100.0


def test_calibration_lut_rejects_mismatched_gray_levels():
    photo = _synthetic_engraved_ramp()
    with pytest.raises(ValueError, match="gray_levels"):
        calibration_lut_from_ramp_photo(
            photo, max_depth_um=100.0, n_steps=11, gray_levels=(0, 128, 255),
        )
