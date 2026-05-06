"""Color tempering recipes (MOPA on stainless / titanium).

Profiles may carry a `color_recipes` block keyed by substrate, with each entry
naming a color and providing the laser parameters that produce it.
See PLAN §21.4 for the schema.

This module exposes a small dataclass plus parser. The recipes are consumed
later by the LightBurn `.clb` writer (Phase 3b) and by the in-app color zone
selector. Today they're only parsed and validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping


_RECIPE_KEYS_NUMERIC = (
    "freq_khz", "pulse_ns", "speed", "power", "line_interval", "passes",
)


@dataclass
class ColorRecipe:
    name: str
    substrate: str
    freq_khz: float
    pulse_ns: float
    speed: float
    power: float
    line_interval: float
    passes: int = 1
    extras: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "substrate": self.substrate,
            "freq_khz": self.freq_khz,
            "pulse_ns": self.pulse_ns,
            "speed": self.speed,
            "power": self.power,
            "line_interval": self.line_interval,
            "passes": self.passes,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d


class ColorRecipeError(ValueError):
    """Raised for malformed color_recipes blocks."""


def _coerce_recipe(name: str, substrate: str, payload: Mapping[str, object]) -> ColorRecipe:
    missing = [k for k in _RECIPE_KEYS_NUMERIC if k not in payload]
    # `passes` defaults to 1 if absent; everything else is required.
    missing = [k for k in missing if k != "passes"]
    if missing:
        raise ColorRecipeError(
            f"color_recipes.{substrate}.{name}: missing required keys {missing}"
        )
    extras = {k: v for k, v in payload.items() if k not in _RECIPE_KEYS_NUMERIC}
    try:
        return ColorRecipe(
            name=name,
            substrate=substrate,
            freq_khz=float(payload["freq_khz"]),
            pulse_ns=float(payload["pulse_ns"]),
            speed=float(payload["speed"]),
            power=float(payload["power"]),
            line_interval=float(payload["line_interval"]),
            passes=int(payload.get("passes", 1)),
            extras=extras,
        )
    except (TypeError, ValueError) as exc:
        raise ColorRecipeError(
            f"color_recipes.{substrate}.{name}: bad numeric value ({exc})"
        ) from exc


def recipes_from_profile(profile_data: Mapping[str, object]) -> List[ColorRecipe]:
    """Parse and validate the color_recipes block. Returns [] if absent."""
    raw = profile_data.get("color_recipes")
    if not raw:
        return []
    if not isinstance(raw, Mapping):
        raise ColorRecipeError("profile.color_recipes must be a mapping")

    recipes: List[ColorRecipe] = []
    for substrate, by_name in raw.items():
        if not isinstance(by_name, Mapping):
            raise ColorRecipeError(
                f"color_recipes.{substrate} must be a mapping of name -> params"
            )
        for color_name, params in by_name.items():
            if not isinstance(params, Mapping):
                raise ColorRecipeError(
                    f"color_recipes.{substrate}.{color_name} must be a mapping"
                )
            recipes.append(_coerce_recipe(str(color_name), str(substrate), params))
    return recipes


def recipe_names(recipes: List[ColorRecipe]) -> List[str]:
    """Stable display names like 'stainless › royal_blue'."""
    return [f"{r.substrate} › {r.name}" for r in recipes]
