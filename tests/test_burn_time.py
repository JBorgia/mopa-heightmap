"""Tests for :mod:`zoedepth.laser.burn_time`."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.burn_time import (
    BurnEstimate,
    DEFAULT_PASS_COUNT,
    PassBurnEstimate,
    estimate_burn_time,
    format_seconds,
)
from zoedepth.laser.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    load_lightburn_card,
)
from zoedepth.laser.stages import (
    PASS_KIND_FORM,
    PASS_KIND_PRE_CLEAN,
    plan_passes,
)


def _ring_heightmap() -> np.ndarray:
    h, w = 64, 64
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = 32.0, 32.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    out = np.full((h, w), 1.0, dtype=np.float32)
    out[(r >= 12) & (r <= 24)] = 0.4
    return out


def _solid_disk_heightmap(h: int = 96, w: int = 96, radius: int = 24) -> np.ndarray:
    """Solid disk subject so the cleanup ring is unambiguously thinner."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    out = np.full((h, w), 1.0, dtype=np.float32)
    out[r <= radius] = 0.3
    return out


def _profile():
    return load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")


# ----------------------------------------------------------- format_seconds

def test_format_seconds_short():
    assert format_seconds(45.5) == "45.5 s"


def test_format_seconds_minutes():
    assert format_seconds(125) == "2 m 05 s"


def test_format_seconds_hours():
    assert format_seconds(3725) == "1 h 02 m 05 s"


def test_format_seconds_zero_or_negative():
    assert format_seconds(0) == "0.0 s"
    assert format_seconds(-5) == "0 s"


# ----------------------------------------------------------- estimate

def test_estimate_burn_time_returns_per_pass_and_total():
    profile = _profile()
    plan = plan_passes(
        heightmap=_ring_heightmap(), profile=profile,
        user_toggles={PASS_KIND_PRE_CLEAN: True},
    )
    est = estimate_burn_time(plan, width_mm=50.0, height_mm=50.0)
    assert isinstance(est, BurnEstimate)
    assert len(est.passes) == len(plan.passes)
    for row in est.passes:
        assert isinstance(row, PassBurnEstimate)
        assert row.seconds >= 0.0
        assert 0.0 <= row.active_fraction <= 1.0
        assert row.pass_count == DEFAULT_PASS_COUNT
    # Total = sum of per-pass times.
    expected_total = sum(p.seconds for p in est.passes)
    assert est.total_seconds == pytest.approx(expected_total, rel=1e-6)


def test_estimate_burn_time_pass_count_overrides_multiply_seconds():
    profile = _profile()
    plan = plan_passes(heightmap=_ring_heightmap(), profile=profile)
    one = estimate_burn_time(plan, width_mm=50.0, height_mm=50.0)
    many = estimate_burn_time(
        plan, width_mm=50.0, height_mm=50.0,
        pass_count_overrides={"form": 32},
    )
    one_form = next(p.seconds for p in one.passes if p.kind == PASS_KIND_FORM)
    many_form = next(p.seconds for p in many.passes if p.kind == PASS_KIND_FORM)
    assert many_form == pytest.approx(32 * one_form, rel=1e-6)


def test_estimate_burn_time_active_fraction_reflects_mask_coverage():
    """A masked color pass over a solid-disk subject covers a smaller fraction than the full-frame form."""
    hm = _solid_disk_heightmap()
    profile = _profile()
    target = profile.entries[2].name
    color_mask = np.zeros_like(hm)
    yy, xx = np.mgrid[: hm.shape[0], : hm.shape[1]].astype(np.float32)
    cy, cx = hm.shape[0] / 2.0, hm.shape[1] / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    color_mask[(r >= 22) & (r <= 24)] = 1.0  # thin ring
    plan = plan_passes(
        heightmap=hm, profile=profile,
        mask_per_color={target: color_mask},
    )
    est = estimate_burn_time(plan, width_mm=50.0, height_mm=50.0)
    form = next(p for p in est.passes if p.kind == PASS_KIND_FORM)
    color = next(p for p in est.passes if p.kind.startswith("color:"))
    # Form covers the whole frame (mask all-ones); the color ring is thin.
    assert color.active_fraction < form.active_fraction


def test_estimate_burn_time_rejects_non_positive_dims():
    profile = _profile()
    plan = plan_passes(heightmap=_ring_heightmap(), profile=profile)
    with pytest.raises(ValueError, match="positive"):
        estimate_burn_time(plan, width_mm=0.0, height_mm=50.0)
    with pytest.raises(ValueError, match="positive"):
        estimate_burn_time(plan, width_mm=50.0, height_mm=-1.0)
