"""Tests for profile authoring (scaffold_profile + extended schema)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mopa.profiles import (
    ProfileValidationError,
    USER_PROFILES_ENV,
    load_profile,
    scaffold_profile,
    validate_profile,
    write_lut_to_profile,
)
from mopa.service import DEFAULT_SETTINGS, merge_profile_settings
from mopa.lut import CalibrationLUT


@pytest.fixture
def user_profiles_dir(tmp_path: Path, monkeypatch):
    target = tmp_path / "profiles"
    monkeypatch.setenv(USER_PROFILES_ENV, str(target))
    return target


def test_scaffold_writes_yaml_in_user_dir(user_profiles_dir: Path):
    settings = dict(DEFAULT_SETTINGS)
    settings["gamma"] = 0.6
    settings["input_clahe"] = True
    path = scaffold_profile("test_brass", settings)
    assert path.exists()
    assert path.parent == user_profiles_dir

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data["name"] == "test_brass"
    assert data["machine"] == "JPT MOPA fiber"
    assert "lightburn_starting_point" in data
    # Only customized values should appear under heightmap.
    assert data["heightmap"]["gamma"] == 0.6
    assert data["heightmap"]["input_clahe"] is True
    assert "contrast" not in data["heightmap"]  # default, omitted


def test_scaffold_round_trips_through_load_profile(user_profiles_dir: Path):
    settings = dict(DEFAULT_SETTINGS)
    settings["gamma"] = 0.55
    settings["edge_refine"] = True
    settings["edge_refine_diameter"] = 7
    scaffold_profile("rt", settings)

    loaded = load_profile("rt")
    assert loaded["heightmap"]["gamma"] == 0.55
    merged = merge_profile_settings(loaded, None)
    assert merged["gamma"] == 0.55
    assert merged["edge_refine"] is True
    assert merged["edge_refine_diameter"] == 7


def test_scaffold_refuses_overwrite_unless_flagged(user_profiles_dir: Path):
    scaffold_profile("dup", dict(DEFAULT_SETTINGS))
    with pytest.raises(FileExistsError):
        scaffold_profile("dup", dict(DEFAULT_SETTINGS))
    # With flag it succeeds.
    scaffold_profile("dup", dict(DEFAULT_SETTINGS), overwrite=True)


def test_scaffold_rejects_dangerous_names(user_profiles_dir: Path):
    for bad in ("..", "with/slash", "x:y", ""):
        with pytest.raises(ValueError):
            scaffold_profile(bad, dict(DEFAULT_SETTINGS))


def test_scaffold_validates_before_writing(user_profiles_dir: Path):
    bad = dict(DEFAULT_SETTINGS)
    bad["gamma"] = 99.0  # out of range
    with pytest.raises(ProfileValidationError):
        scaffold_profile("bad", bad)
    assert not (user_profiles_dir / "bad.yaml").exists()


def test_extended_schema_accepts_phase2_keys():
    payload = {
        "name": "ok",
        "heightmap": {
            "gamma": 0.7,
            "input_clahe": True,
            "input_clahe_clip": 3.0,
            "edge_refine": True,
            "edge_refine_sigma_color": 0.1,
            "dither": True,
            "dither_levels": 64,
        },
    }
    validate_profile(payload)  # no exception


def test_extended_schema_still_rejects_unknown_heightmap_key():
    payload = {"name": "x", "heightmap": {"nope": 1}}
    with pytest.raises(ProfileValidationError):
        validate_profile(payload)


def test_write_lut_to_profile_round_trips(user_profiles_dir):
    scaffold_profile("brass_cal", dict(DEFAULT_SETTINGS))
    depths = [0.0, 4.0, 9.0, 14.0, 19.0, 26.0, 33.0, 41.0, 50.0, 58.0, 66.0]
    lut = CalibrationLUT.from_measurements(depths, note="unit test")
    path = write_lut_to_profile("brass_cal", lut.to_dict())
    assert path.exists()
    with path.open("r", encoding="utf-8") as h:
        data = yaml.safe_load(h)
    assert "calibration_lut" in data
    samples = data["calibration_lut"]["samples"]
    assert len(samples) == 11
    # Last patch is the brightest gray and matches the deepest depth.
    assert samples[-1][1] == pytest.approx(66.0)
    assert data["calibration_lut"]["max_depth_um"] == pytest.approx(66.0)


def test_write_lut_preserves_other_profile_keys(user_profiles_dir):
    scaffold_profile("preserve", dict(DEFAULT_SETTINGS))
    before = load_profile("preserve")
    lut = CalibrationLUT.from_measurements([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    write_lut_to_profile("preserve", lut.to_dict())
    after = load_profile("preserve")
    # All non-calibration keys survive.
    for key in before:
        if key in {"__profile_path__", "calibration_lut"}:
            continue
        assert after.get(key) == before.get(key), f"key {key} changed"

