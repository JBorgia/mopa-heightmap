"""Tests for the LightBurn ``.lbrn2`` writer.

The contract is round-trip fidelity: parsing the writer's output with the
existing :func:`zoedepth.laser.lightburn_cards.load_lightburn_card` parser
must recover ``ColorEntry`` rows whose raw payload matches the source card
verbatim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zoedepth.laser.lbrn_writer import (
    LBRN_DEFAULT_APP_VERSION,
    LBRN_DEFAULT_MIRROR_X,
    LBRN_DEFAULT_MIRROR_Y,
    LBRN_DEFAULT_MATERIAL_HEIGHT,
    LBRN_FORMAT_VERSION,
    LBRN_IDENTITY_XFORM,
    ShapeRef,
    build_lbrn_tree,
    write_lbrn,
)
from zoedepth.laser.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    load_lightburn_card,
)
from zoedepth.laser.stages import (
    DEFAULT_PASS_ORDER,
    PASS_KIND_FORM,
    plan_passes,
)


def _profile():
    return load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")


def _heightmap():
    return np.linspace(0.0, 1.0, 8 * 8, dtype=np.float32).reshape(8, 8)


# ----------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert LBRN_FORMAT_VERSION == "1"
    assert LBRN_DEFAULT_APP_VERSION == "1.2.04"
    assert LBRN_DEFAULT_MIRROR_X == "False"
    assert LBRN_DEFAULT_MIRROR_Y == "False"
    assert LBRN_DEFAULT_MATERIAL_HEIGHT == 0.0
    assert LBRN_IDENTITY_XFORM == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


# ----------------------------------------------------------- builder

def test_build_lbrn_tree_rejects_duplicate_indices():
    profile = _profile()
    e0 = profile.entries[0]
    with pytest.raises(ValueError, match="Duplicate"):
        build_lbrn_tree(entries=[e0, e0], shapes=[])


def test_build_lbrn_tree_rejects_shape_with_unknown_cut_index():
    profile = _profile()
    with pytest.raises(ValueError, match="no CutSetting"):
        build_lbrn_tree(
            entries=[profile.entries[0]],
            shapes=[ShapeRef(cut_index=999)],
        )


def test_build_lbrn_tree_sets_root_attribs():
    profile = _profile()
    tree = build_lbrn_tree(
        entries=[profile.entries[0]],
        shapes=[ShapeRef(cut_index=profile.entries[0].index)],
        material_height=2.5,
        mirror_x=True,
        mirror_y=False,
    )
    root = tree.getroot()
    assert root.tag == "LightBurnProject"
    assert root.attrib["FormatVersion"] == LBRN_FORMAT_VERSION
    assert root.attrib["MaterialHeight"].startswith("2.5")
    assert root.attrib["MirrorX"] == "True"
    assert root.attrib["MirrorY"] == LBRN_DEFAULT_MIRROR_Y


# ------------------------------------------------------ write + round-trip

def test_write_lbrn_round_trips_through_parser(tmp_path: Path):
    profile = _profile()
    plan = plan_passes(heightmap=_heightmap(), profile=profile)
    # Fake per-pass PNG paths just to exercise the SourceFile encoding.
    pngs = {p.id: tmp_path / f"{p.id}.png" for p in plan.passes}
    for png in pngs.values():
        png.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
    out = tmp_path / "project.lbrn2"
    write_lbrn(out, plan, pass_pngs=pngs)
    reparsed = load_lightburn_card(out)

    written_indices = {ep.cut_setting.index for ep in plan.passes}
    for written_idx in written_indices:
        original = profile.by_index[written_idx]
        round_tripped = reparsed.by_index[written_idx]
        # ``raw`` is the verbatim XML payload; it MUST be byte-identical
        # for every field we wrote (we may have added entries we didn't
        # touch but for the ones that exist on both sides, they match).
        for key, value in original.raw.items():
            assert round_tripped.raw.get(key) == value, (
                f"field {key} drifted on round-trip: "
                f"{value!r} -> {round_tripped.raw.get(key)!r}"
            )


def test_write_lbrn_preserves_app_version_from_source_card(tmp_path: Path):
    profile = _profile()
    plan = plan_passes(heightmap=_heightmap(), profile=profile)
    out = tmp_path / "project.lbrn2"
    write_lbrn(out, plan)
    reparsed = load_lightburn_card(out)
    # The writer's default falls back to the profile's recorded AppVersion
    # so the file looks indistinguishable from a hand-authored one.
    assert reparsed.app_version == profile.app_version


def test_write_lbrn_appends_extra_entries(tmp_path: Path):
    profile = _profile()
    plan = plan_passes(
        heightmap=_heightmap(), profile=profile,
        user_toggles={k: False for k in DEFAULT_PASS_ORDER if k != PASS_KIND_FORM},
    )
    extras = [profile.entries[-1], profile.entries[-2]]
    out = tmp_path / "project.lbrn2"
    write_lbrn(out, plan, extra_entries=extras)
    reparsed = load_lightburn_card(out)
    expected = {ep.cut_setting.index for ep in plan.passes} | {e.index for e in extras}
    assert set(reparsed.by_index.keys()) == expected


def test_write_lbrn_uses_relative_png_paths(tmp_path: Path):
    profile = _profile()
    plan = plan_passes(heightmap=_heightmap(), profile=profile)
    png = tmp_path / "form.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "deep" / "project.lbrn2"
    write_lbrn(out, plan, pass_pngs={PASS_KIND_FORM: png})
    text = out.read_text(encoding="utf-8")
    # The bitmap shape must reference the PNG by path relative to the
    # project file, not by absolute path (so the bundle stays portable).
    assert "form.png" in text
    assert str(tmp_path) not in text or text.count(str(tmp_path)) == 0
