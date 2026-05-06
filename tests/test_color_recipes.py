"""Tests for color_recipes parsing."""
from __future__ import annotations

import pytest

from mopa.color_recipes import (
    ColorRecipe,
    ColorRecipeError,
    recipe_names,
    recipes_from_profile,
)


def _profile():
    return {
        "color_recipes": {
            "stainless": {
                "deep_blue": {
                    "freq_khz": 60, "pulse_ns": 4, "speed": 800,
                    "power": 35, "line_interval": 0.04, "passes": 1,
                },
                "gold": {
                    "freq_khz": 200, "pulse_ns": 20, "speed": 1500,
                    "power": 25, "line_interval": 0.04,  # passes default
                },
            },
            "titanium": {
                "purple": {
                    "freq_khz": 100, "pulse_ns": 8, "speed": 1200,
                    "power": 28, "line_interval": 0.04, "passes": 1,
                },
            },
        }
    }


def test_recipes_from_profile_parses_all():
    recipes = recipes_from_profile(_profile())
    assert len(recipes) == 3
    names = {(r.substrate, r.name) for r in recipes}
    assert names == {("stainless", "deep_blue"), ("stainless", "gold"),
                     ("titanium", "purple")}


def test_default_passes_is_one():
    recipes = recipes_from_profile(_profile())
    gold = next(r for r in recipes if r.name == "gold")
    assert gold.passes == 1


def test_missing_required_key_raises():
    bad = {
        "color_recipes": {
            "stainless": {
                "broken": {"freq_khz": 60, "pulse_ns": 4},  # missing speed/power/etc.
            }
        }
    }
    with pytest.raises(ColorRecipeError):
        recipes_from_profile(bad)


def test_non_mapping_recipes_raises():
    with pytest.raises(ColorRecipeError):
        recipes_from_profile({"color_recipes": "not a mapping"})


def test_extras_preserved():
    bad_payload = {
        "color_recipes": {
            "stainless": {
                "x": {
                    "freq_khz": 60, "pulse_ns": 4, "speed": 800,
                    "power": 35, "line_interval": 0.04,
                    "_note": "for jewelry only",
                },
            }
        }
    }
    recipes = recipes_from_profile(bad_payload)
    assert recipes[0].extras == {"_note": "for jewelry only"}


def test_recipe_to_dict_round_trips_basic_fields():
    r = ColorRecipe(
        name="x", substrate="stainless",
        freq_khz=60, pulse_ns=4, speed=800,
        power=35, line_interval=0.04, passes=2,
    )
    d = r.to_dict()
    assert d["passes"] == 2
    assert d["substrate"] == "stainless"


def test_recipe_names_format():
    recipes = recipes_from_profile(_profile())
    names = recipe_names(recipes)
    assert all("›" in n for n in names)


def test_empty_profile_returns_empty_list():
    assert recipes_from_profile({}) == []
    assert recipes_from_profile({"name": "x"}) == []
