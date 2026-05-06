import pytest
import yaml

from mopa.profiles import (
    ProfileValidationError,
    list_profiles,
    load_profile,
    validate_profile,
)


@pytest.mark.parametrize("name", ["mopa_60w_brass", "mopa_60w_stainless", "mopa_60w_aluminum", "mopa_60w_copper"])
def test_shipped_profiles_load_and_validate(name):
    data = load_profile(name)
    assert data["name"]
    assert "heightmap" in data


def test_list_profiles_returns_at_least_the_shipped_four():
    names = set(list_profiles())
    assert {"mopa_60w_brass", "mopa_60w_stainless", "mopa_60w_aluminum", "mopa_60w_copper"} <= names


def test_unknown_top_level_key_rejected():
    bad = {"name": "x", "heightmap": {}, "wat": True}
    with pytest.raises(ProfileValidationError) as exc:
        validate_profile(bad)
    assert "wat" in str(exc.value)


def test_out_of_range_value_rejected():
    bad = {"name": "x", "heightmap": {"gamma": 99.0}}
    with pytest.raises(ProfileValidationError) as exc:
        validate_profile(bad)
    assert "gamma" in str(exc.value)


def test_far_must_exceed_near():
    bad = {"name": "x", "heightmap": {"near_percentile": 90.0, "far_percentile": 80.0}}
    with pytest.raises(ProfileValidationError):
        validate_profile(bad)


def test_unknown_smooth_mode_rejected():
    bad = {"name": "x", "heightmap": {"smooth": "magic"}}
    with pytest.raises(ProfileValidationError):
        validate_profile(bad)


def test_load_profile_from_path(tmp_path):
    p = tmp_path / "custom.yaml"
    p.write_text(yaml.safe_dump({
        "name": "custom",
        "black_is_deep": True,
        "heightmap": {"gamma": 0.9},
    }), encoding="utf-8")
    data = load_profile(str(p))
    assert data["name"] == "custom"
    assert data["heightmap"]["gamma"] == 0.9
