"""Tests for the engraving pass planner."""
from __future__ import annotations

import numpy as np
import pytest

from mopa.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    load_lightburn_card,
)
from mopa.stages import (
    DEFAULT_PASS_ORDER,
    PASS_KIND_COLOR_PREFIX,
    PASS_KIND_FORM,
    PASS_KIND_PHOTO_TONAL,
    PASS_KIND_PRE_CLEAN,
    PASS_KIND_SIGNATURE,
    EngravingPass,
    PassPlan,
    plan_passes,
)


def _profile():
    return load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")


def _heightmap():
    return np.linspace(0.0, 1.0, 8 * 8, dtype=np.float32).reshape(8, 8)


# ----------------------------------------------------------- pass-kind constants

def test_default_pass_order_form_runs_before_refinements():
    """Form (the depth layer) precedes photo-tonal and signature."""
    order = DEFAULT_PASS_ORDER
    assert order.index(PASS_KIND_FORM) < order.index(PASS_KIND_PHOTO_TONAL)
    assert order.index(PASS_KIND_FORM) < order.index(PASS_KIND_SIGNATURE)


def test_color_prefix_is_documented():
    assert PASS_KIND_COLOR_PREFIX == "color:"


# ----------------------------------------------------------- planner contract

def test_plan_with_default_toggles_emits_only_form():
    """Refinement passes are opt-in; default plan ships just the depth layer."""
    plan = plan_passes(heightmap=_heightmap(), profile=_profile())
    assert isinstance(plan, PassPlan)
    kinds = [p.kind for p in plan.passes]
    assert kinds == [PASS_KIND_FORM]


def test_pre_clean_opt_in_via_toggle():
    plan = plan_passes(
        heightmap=_heightmap(), profile=_profile(),
        user_toggles={PASS_KIND_PRE_CLEAN: True},
    )
    kinds = [p.kind for p in plan.passes]
    assert PASS_KIND_PRE_CLEAN in kinds
    assert PASS_KIND_FORM in kinds


def test_form_can_be_disabled_via_toggle():
    """Edge case — the planner allows form=False if a caller really wants it."""
    plan = plan_passes(
        heightmap=_heightmap(), profile=_profile(),
        user_toggles={PASS_KIND_FORM: False},
    )
    assert PASS_KIND_FORM not in [p.kind for p in plan.passes]


def test_color_passes_inserted_in_profile_index_order():
    profile = _profile()
    h = _heightmap()
    selected = [profile.entries[1].name, profile.entries[3].name, profile.entries[2].name]
    masks = {name: np.ones_like(h) for name in selected}
    plan = plan_passes(
        heightmap=h, profile=profile,
        user_toggles={PASS_KIND_FORM: False},
        mask_per_color=masks,
    )
    indices = [p.cut_setting.index for p in plan.passes]
    assert indices == sorted(indices)


def test_color_pass_can_be_individually_toggled_off():
    profile = _profile()
    target = profile.entries[2].name
    plan = plan_passes(
        heightmap=_heightmap(), profile=profile,
        user_toggles={f"{PASS_KIND_COLOR_PREFIX}{target}": False},
        mask_per_color={target: np.ones((8, 8), dtype=np.float32)},
    )
    assert all(p.name != target or p.kind != f"{PASS_KIND_COLOR_PREFIX}{target}"
               for p in plan.passes)


def test_kind_color_overrides_redirect_pass_to_different_card_row():
    profile = _profile()
    alternative = profile.entries[5].name
    plan = plan_passes(
        heightmap=_heightmap(), profile=profile,
        kind_color_overrides={PASS_KIND_FORM: alternative},
    )
    forms = plan.by_kind(PASS_KIND_FORM)
    assert len(forms) == 1
    assert forms[0].cut_setting.name == alternative


def test_kind_color_overrides_support_noncanonical_form_layer_names():
    profile = _profile()
    renamed = profile.entries[5]
    replacement = type(renamed)(
        index=renamed.index,
        name="CustomDepth",
        max_power=renamed.max_power,
        speed=renamed.speed,
        frequency=renamed.frequency,
        q_pulse_width=renamed.q_pulse_width,
        interval=renamed.interval,
        min_power=renamed.min_power,
        max_power_2=renamed.max_power_2,
        priority=renamed.priority,
        flood_fill=renamed.flood_fill,
        bidir=renamed.bidir,
        raw=dict(renamed.raw),
    )
    custom_profile = type(profile)(
        name=profile.name,
        source_path=profile.source_path,
        machine_label=profile.machine_label,
        wattage=profile.wattage,
        app_version=profile.app_version,
        entries=[replacement if entry.index == renamed.index else entry for entry in profile.entries],
        thumbnail_b64=profile.thumbnail_b64,
    )
    plan = plan_passes(
        heightmap=_heightmap(),
        profile=custom_profile,
        kind_color_overrides={PASS_KIND_FORM: "CustomDepth"},
    )
    forms = plan.by_kind(PASS_KIND_FORM)
    assert len(forms) == 1
    assert forms[0].cut_setting.name == "CustomDepth"


def test_pass_dropped_silently_when_color_name_missing():
    profile = _profile()
    plan = plan_passes(
        heightmap=_heightmap(), profile=profile,
        kind_color_overrides={PASS_KIND_FORM: "Z99_does_not_exist"},
    )
    assert plan.by_kind(PASS_KIND_FORM) == ()


def test_planner_rejects_non_2d_heightmap():
    with pytest.raises(ValueError, match="2-D"):
        plan_passes(heightmap=np.zeros((4, 4, 3), dtype=np.float32), profile=_profile())


def test_planner_validates_per_kind_mask_shape():
    profile = _profile()
    with pytest.raises(ValueError, match="expected"):
        plan_passes(
            heightmap=_heightmap(), profile=profile,
            masks={PASS_KIND_FORM: np.zeros((4, 4), dtype=np.float32)},
        )


def test_engraving_pass_is_immutable_dataclass():
    profile = _profile()
    plan = plan_passes(heightmap=_heightmap(), profile=profile)
    p = plan.passes[0]
    assert isinstance(p, EngravingPass)
    with pytest.raises(Exception):
        p.enabled = False  # frozen dataclass


def test_pass_plan_lookup_helpers():
    plan = plan_passes(heightmap=_heightmap(), profile=_profile())
    p = plan.by_id(PASS_KIND_FORM)
    assert p is not None and p.kind == PASS_KIND_FORM
    assert plan.by_id("not_a_real_pass") is None
