"""Tests for the LightBurn ``.clb`` Cut Library writer.

Round-trip: write_clb → load_lightburn_card on the result must read the
exact same ColorEntry rows back. (load_lightburn_card accepts any XML
whose root contains ``<CutSetting>`` children, so .clb files round-trip
through the same loader as .lbrn2 projects.)
"""
from __future__ import annotations

from pathlib import Path

import xml.etree.ElementTree as ET

from zoedepth.laser.lbrn_writer import CLB_ROOT_TAG, build_clb_tree, write_clb
from zoedepth.laser.lightburn_cards import (
    DEFAULT_CARDS_DIR, DEFAULT_PROFILE_NAME, load_lightburn_card,
)


def _profile():
    return load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")


def test_clb_root_tag_constant():
    assert CLB_ROOT_TAG == "CutSettings"


def test_build_clb_tree_root_is_cut_settings():
    profile = _profile()
    tree = build_clb_tree(profile.entries[:3])
    root = tree.getroot()
    assert root.tag == CLB_ROOT_TAG
    cuts = root.findall("CutSetting")
    assert len(cuts) == 3


def test_build_clb_tree_rejects_duplicate_indices():
    profile = _profile()
    e0 = profile.entries[0]
    import pytest
    with pytest.raises(ValueError, match="Duplicate"):
        build_clb_tree([e0, e0])


def test_write_clb_round_trips_through_card_loader(tmp_path: Path):
    profile = _profile()
    out = tmp_path / "library.clb"
    write_clb(out, profile.entries[:5])

    # Re-parse with the same loader the importer uses for cards.
    reparsed = load_lightburn_card(out)
    for source_entry in profile.entries[:5]:
        round_tripped = reparsed.by_index[source_entry.index]
        for key, value in source_entry.raw.items():
            assert round_tripped.raw.get(key) == value, (
                f"field {key} drifted on .clb round-trip"
            )


def test_write_clb_creates_parent_directory(tmp_path: Path):
    profile = _profile()
    out = tmp_path / "deep" / "nested" / "library.clb"
    write_clb(out, profile.entries[:2])
    assert out.exists()
    # And it parses as XML with the right root.
    root = ET.parse(out).getroot()
    assert root.tag == CLB_ROOT_TAG
