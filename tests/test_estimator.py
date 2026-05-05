"""Tests for the engraving-time estimator."""
from __future__ import annotations

import pytest

from zoedepth.laser.estimator import (
    EngraveEstimate,
    estimate_engrave_time,
    estimate_from_profile,
)


def test_estimate_basic_arithmetic():
    # 100 mm tall, 100 mm wide, 0.1 mm interval -> 1000 lines * 100 mm = 100_000 mm.
    # At 1000 mm/s with 1 pass and overhead 1.0 -> 100 s.
    est = estimate_engrave_time(
        image_height_px=1000, image_width_px=1000,
        physical_height_mm=100.0, physical_width_mm=100.0,
        lightburn_starting_point={"speed": 1000.0, "passes": 1, "line_interval": 0.1},
        overhead_factor=1.0,
    )
    assert isinstance(est, EngraveEstimate)
    assert est.line_count == 1000
    assert est.seconds == pytest.approx(100.0, rel=0.01)


def test_estimate_passes_multiply_time():
    base = estimate_engrave_time(
        100, 100,
        physical_height_mm=50.0, physical_width_mm=50.0,
        lightburn_starting_point={"speed": 2000.0, "passes": 1, "line_interval": 0.05},
        overhead_factor=1.0,
    )
    multi = estimate_engrave_time(
        100, 100,
        physical_height_mm=50.0, physical_width_mm=50.0,
        lightburn_starting_point={"speed": 2000.0, "passes": 4, "line_interval": 0.05},
        overhead_factor=1.0,
    )
    assert multi.seconds == pytest.approx(base.seconds * 4, rel=0.001)
    assert multi.line_count == base.line_count * 4


def test_estimate_overhead_factor_applied():
    est = estimate_engrave_time(
        100, 100,
        physical_height_mm=10.0, physical_width_mm=10.0,
        lightburn_starting_point={"speed": 1000.0, "passes": 1, "line_interval": 0.1},
        overhead_factor=1.5,
    )
    # 100 lines * 10 mm / 1000 = 1.0 s, * 1.5 = 1.5 s.
    assert est.seconds == pytest.approx(1.5, rel=0.001)


def test_estimate_invalid_speed_raises():
    with pytest.raises(ValueError):
        estimate_engrave_time(
            10, 10,
            physical_height_mm=10.0, physical_width_mm=10.0,
            lightburn_starting_point={"speed": 0, "passes": 1, "line_interval": 0.1},
        )


def test_estimate_invalid_interval_raises():
    with pytest.raises(ValueError):
        estimate_engrave_time(
            10, 10,
            physical_height_mm=10.0, physical_width_mm=10.0,
            lightburn_starting_point={"speed": 1000, "passes": 1, "line_interval": 0},
        )


def test_estimate_long_job_adds_warning_note():
    est = estimate_engrave_time(
        2000, 2000,
        physical_height_mm=200.0, physical_width_mm=200.0,
        lightburn_starting_point={"speed": 500.0, "passes": 5, "line_interval": 0.04},
    )
    assert any("1 hour" in note for note in est.notes)


def test_estimate_from_profile_uses_cut_block():
    profile = {
        "lightburn_starting_point": {
            "speed": 2000, "passes": 2, "line_interval": 0.05,
        }
    }
    est = estimate_from_profile((1000, 1000), (50.0, 50.0), profile)
    assert est.line_count == 2000


def test_estimate_human_format():
    est = estimate_engrave_time(
        100, 100,
        physical_height_mm=10.0, physical_width_mm=10.0,
        lightburn_starting_point={"speed": 1000.0, "passes": 1, "line_interval": 0.1},
        overhead_factor=1.0,
    )
    s = est.human()
    assert isinstance(s, str) and "s" in s


def test_estimate_accepts_unit_suffixed_keys_from_shipped_profiles():
    """Shipped MOPA YAMLs use `speed_mm_s` and `line_interval_mm` instead of
    the bare names. The estimator must accept both forms.
    """
    est = estimate_engrave_time(
        1000, 1000,
        physical_height_mm=100.0, physical_width_mm=100.0,
        lightburn_starting_point={
            "speed_mm_s": 1000.0,
            "passes": 1,
            "line_interval_mm": 0.1,
        },
        overhead_factor=1.0,
    )
    # Same arithmetic as test_estimate_basic_arithmetic.
    assert est.line_count == 1000
    assert est.seconds == pytest.approx(100.0, rel=0.01)


def test_estimate_against_shipped_brass_profile_runs():
    from zoedepth.laser.profiles import load_profile

    profile = load_profile("mopa_60w_brass")
    est = estimate_from_profile((1000, 1000), (50.0, 50.0), profile)
    assert est.seconds > 0
    assert est.line_count > 0
