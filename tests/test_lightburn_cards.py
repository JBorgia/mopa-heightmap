"""Tests for the LightBurn ``.lbrn2`` color-card importer."""
from __future__ import annotations

from pathlib import Path

import pytest

from mopa.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    ColorEntry,
    MaterialProfile,
    load_all_cards,
    load_lightburn_card,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS = REPO_ROOT / "LightBurn Colour Card"
EXPECTED_CARDS = {
    "Colour20W-M7", "Colour30W-M7", "Colour60W-M7",
    "Colour80W-M7", "Colour100W-M7",
}


def test_default_cards_dir_resolves_to_repo_folder():
    assert DEFAULT_CARDS_DIR == CARDS


@pytest.mark.parametrize("stem", sorted(EXPECTED_CARDS))
def test_each_supplied_card_parses(stem: str):
    profile = load_lightburn_card(CARDS / f"{stem}.lbrn2")
    assert isinstance(profile, MaterialProfile)
    assert profile.name == stem
    assert len(profile.entries) >= 8, "color card should have at least 8 entries"
    # Wattage must be parsed from the filename.
    assert profile.wattage is not None
    assert str(profile.wattage) in stem


def test_60w_card_is_default_profile():
    profile = load_lightburn_card(CARDS / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    assert profile.wattage == 60


def test_color_entries_have_required_machine_params():
    profile = load_lightburn_card(CARDS / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    for entry in profile.entries:
        assert isinstance(entry, ColorEntry)
        assert entry.name.startswith("C")
        # Power 0..100 %, but a few cards push to 100 inclusive.
        assert 0 <= entry.max_power <= 100
        assert entry.speed > 0
        # MOPA frequency band: 1 kHz .. 4 MHz comfortably covers all cards.
        assert 1_000 <= entry.frequency <= 4_000_000
        # Q-pulse width is in ns; LightBurn caps at 200.
        assert 1 <= entry.q_pulse_width <= 200
        # Line interval in mm — anything in [1um .. 1mm] is plausible.
        assert 0.0005 <= entry.interval <= 1.0


def test_indices_unique_and_lookups_work():
    profile = load_lightburn_card(CARDS / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    indices = [e.index for e in profile.entries]
    assert len(set(indices)) == len(indices)
    # by_index round-trip
    for entry in profile.entries:
        assert profile.by_index[entry.index] is entry
        assert profile.by_name[entry.name] is entry


def test_load_all_cards_returns_all_supplied():
    profiles = load_all_cards()
    assert set(profiles.keys()) == EXPECTED_CARDS
    # Each profile is keyed by its filename stem.
    for stem, profile in profiles.items():
        assert profile.name == stem


def test_missing_card_raises_filenotfound(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_lightburn_card(tmp_path / "does_not_exist.lbrn2")


def test_malformed_card_raises_valueerror(tmp_path: Path):
    bad = tmp_path / "bad.lbrn2"
    bad.write_text("<not-a-lightburn-project/>")
    with pytest.raises(ValueError, match="not a LightBurn project"):
        load_lightburn_card(bad)


def test_load_all_cards_handles_missing_dir(tmp_path: Path):
    assert load_all_cards(tmp_path / "nope") == {}
