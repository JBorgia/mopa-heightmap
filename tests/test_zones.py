"""Tests for mopa/zones.py — zone mask computation and heightmap compositing."""
from __future__ import annotations

import numpy as np
import pytest

from mopa.zones import (
    VALID_SHAPES,
    ZONE_INDEX,
    entries_from_zone_params,
    zone_hm_for_pass,
    zone_masks_from_geometry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hm(h: int = 64, w: int = 64) -> np.ndarray:
    return np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)


def _geo(shape: str = "circle", **kw) -> dict:
    defaults = dict(
        h=64, w=64,
        print_w_mm=25.0,
        print_h_mm=25.0,
        px_per_mm=64 / 25.0,
    )
    defaults.update(kw)
    return zone_masks_from_geometry(
        defaults.pop("h"), defaults.pop("w"),
        shape,
        **defaults,
    )


# ---------------------------------------------------------------------------
# VALID_SHAPES / ZONE_INDEX constants
# ---------------------------------------------------------------------------

def test_valid_shapes_contains_expected():
    assert {"circle", "rectangle", "hexagon", "triangle", "donut", "shield", "path"} <= VALID_SHAPES


def test_zone_index_device_is_c01():
    assert ZONE_INDEX["zone:device"] == 1


def test_zone_index_field_is_c02():
    assert ZONE_INDEX["zone:field"] == 2


# ---------------------------------------------------------------------------
# zone_masks_from_geometry — return keys and dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["circle", "rectangle", "hexagon", "triangle"])
def test_masks_return_all_keys(shape):
    masks = _geo(shape)
    assert set(masks) == {"outer", "field", "border", "rim", "exergue"}


@pytest.mark.parametrize("shape", ["circle", "rectangle"])
def test_masks_are_float32_in_unit_range(shape):
    masks = _geo(shape)
    for key, arr in masks.items():
        assert arr.dtype == np.float32, f"{key} is not float32"
        assert arr.min() >= 0.0, f"{key} has values below 0"
        assert arr.max() <= 1.0, f"{key} has values above 1"


@pytest.mark.parametrize("shape", ["circle", "rectangle"])
def test_masks_have_correct_shape(shape):
    h, w = 48, 64
    px_per_mm = 48 / 25.0
    masks = zone_masks_from_geometry(h, w, shape, print_w_mm=25.0, print_h_mm=25.0, px_per_mm=px_per_mm)
    for key, arr in masks.items():
        assert arr.shape == (h, w), f"{key} shape mismatch: {arr.shape}"


# ---------------------------------------------------------------------------
# zone_masks_from_geometry — circle geometry
# ---------------------------------------------------------------------------

def test_circle_outer_mask_is_circular():
    masks = _geo("circle")
    outer = masks["outer"]
    cx, cy = outer.shape[1] / 2.0, outer.shape[0] / 2.0
    # Corner pixels (distance >> radius) should be ~0
    assert outer[0, 0] < 0.1
    # Centre pixel should be ~1
    assert outer[int(cy), int(cx)] > 0.9


def test_circle_field_plus_border_plus_rim_leq_outer():
    masks = _geo("circle")
    combined = masks["field"] + masks["border"] + masks["rim"]
    # After feathering, sum can briefly exceed outer at boundaries, but clamp handles it
    assert float(np.max(combined - masks["outer"])) < 0.3


def test_circle_rim_is_annular_ring():
    masks = _geo("circle")
    rim = masks["rim"]
    # Centre of a circle rim should be zero (rim is the outer ring)
    h, w = rim.shape
    assert rim[h // 2, w // 2] < 0.05


def test_exergue_is_bottom_strip():
    masks = _geo("circle")
    ex = masks["exergue"]
    h = ex.shape[0]
    # Top row should be ~0, bottom row should have content
    assert ex[0, :].mean() < 0.05
    assert ex[-1, :].mean() > 0.1


# ---------------------------------------------------------------------------
# zone_masks_from_geometry — donut
# ---------------------------------------------------------------------------

def test_donut_suppresses_exergue():
    masks = _geo("donut", hole_radius_mm=4.0)
    assert masks["exergue"].max() < 0.01


def test_donut_hole_has_zero_outer():
    h, w = 64, 64
    px_per_mm = 64 / 25.0
    hole_r_mm = 4.0
    masks = zone_masks_from_geometry(
        h, w, "donut",
        print_w_mm=25.0, print_h_mm=25.0,
        px_per_mm=px_per_mm,
        hole_radius_mm=hole_r_mm,
    )
    outer = masks["outer"]
    # Centre pixel falls inside the hole
    assert outer[h // 2, w // 2] < 0.1


# ---------------------------------------------------------------------------
# zone_masks_from_geometry — rectangle
# ---------------------------------------------------------------------------

def test_rectangle_outer_is_rectangular():
    h, w = 64, 64
    px_per_mm = 64 / 25.0
    masks = zone_masks_from_geometry(
        h, w, "rectangle",
        print_w_mm=20.0, print_h_mm=20.0,
        px_per_mm=px_per_mm,
    )
    outer = masks["outer"]
    # Centre should be in
    assert outer[h // 2, w // 2] > 0.9
    # True corner should be out (25mm box > 20mm rectangle)
    assert outer[0, 0] < 0.1


# ---------------------------------------------------------------------------
# zone_masks_from_geometry — sigma_scale
# ---------------------------------------------------------------------------

def test_sigma_scale_zero_gives_hard_edges():
    masks_hard = _geo("circle", sigma_scale=0.0)
    masks_soft = _geo("circle", sigma_scale=3.0)
    # Hard-edge outer should have more pixels near 0 or 1
    def _bimodal(arr):
        flat = arr.ravel()
        return float(np.sum((flat < 0.05) | (flat > 0.95))) / flat.size

    assert _bimodal(masks_hard["outer"]) > _bimodal(masks_soft["outer"])


# ---------------------------------------------------------------------------
# zone_hm_for_pass
# ---------------------------------------------------------------------------

def test_zone_hm_full_mask_preserves_heightmap():
    hm = _hm()
    full = np.ones_like(hm)
    result = zone_hm_for_pass(hm, full)
    # With full mask, formula is 1-(1-hm)*1 = hm; bilateral may shift slightly
    np.testing.assert_allclose(result, hm, atol=0.05)


def test_zone_hm_zero_mask_raises_to_surface():
    hm = _hm()
    empty = np.zeros_like(hm)
    result = zone_hm_for_pass(hm, empty)
    # 1-(1-hm)*0 = 1.0 everywhere
    np.testing.assert_allclose(result, np.ones_like(hm), atol=0.01)


def test_zone_hm_output_is_float32_in_unit_range():
    hm = _hm()
    mask = np.ones_like(hm) * 0.5
    result = zone_hm_for_pass(hm, mask)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_zone_hm_output_shape_matches_input():
    hm = _hm(48, 72)
    mask = np.ones((48, 72), dtype=np.float32)
    result = zone_hm_for_pass(hm, mask)
    assert result.shape == (48, 72)


# ---------------------------------------------------------------------------
# entries_from_zone_params
# ---------------------------------------------------------------------------

_STARTING = {
    "speed_mm_s": 600,
    "power_percent": 75,
    "frequency_khz": 60,
    "pulse_width_ns": 130,
    "line_interval_mm": 0.015,
    "passes": 256,
}

_ZONE_PARAMS = {
    "field": {
        "speed_mm_s": 2100,
        "power_percent": 32,
        "frequency_khz": 100,
        "pulse_width_ns": 185,
        "line_interval_mm": 0.02,
        "passes": 2,
    },
    "rim": {
        "speed_mm_s": 1500,
        "power_percent": 28,
        "frequency_khz": 80,
        "pulse_width_ns": 185,
        "line_interval_mm": 0.02,
        "passes": 2,
    },
}


def test_entries_from_zone_params_returns_configured_zones():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    assert "zone:field" in entries
    assert "zone:rim" in entries


def test_entries_device_falls_back_to_starting_point():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    assert "zone:device" in entries
    dev = entries["zone:device"]
    assert dev.speed == float(_STARTING["speed_mm_s"])
    assert dev.max_power == float(_STARTING["power_percent"])


def test_entries_unconfigured_zone_absent():
    # border/exergue not in _ZONE_PARAMS → not returned
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    assert "zone:border" not in entries
    assert "zone:exergue" not in entries


def test_entries_field_uses_correct_slot_index():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    assert entries["zone:field"].index == ZONE_INDEX["zone:field"]


def test_entries_device_uses_slot_1():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    assert entries["zone:device"].index == 1


def test_entries_raw_dict_has_required_keys():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    required = {"index", "name", "maxPower", "speed", "frequency", "QPulseWidth",
                "interval", "numPasses", "ditherMode"}
    for zone_kind, entry in entries.items():
        missing = required - set(entry.raw.keys())
        assert not missing, f"{zone_kind} raw missing: {missing}"


def test_entries_field_speed_matches_zone_params():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING)
    assert entries["zone:field"].speed == float(_ZONE_PARAMS["field"]["speed_mm_s"])


def test_entries_cleanup_pass_set_on_device():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING, cleanup_every_passes=18)
    assert entries["zone:device"].raw.get("cleanupPass") == "1"


def test_entries_cleanup_pass_not_on_field():
    entries = entries_from_zone_params(_ZONE_PARAMS, _STARTING, cleanup_every_passes=18)
    assert "cleanupPass" not in entries["zone:field"].raw
