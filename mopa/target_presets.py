"""Target-object presets — coin / ring / pendant / plaque / portrait.

A target preset bundles "this is what I'm engraving" defaults: physical
print dimensions, polarity-invert default, and a starter ``HeightmapSettings``
override block. Composing a CLI / API call with ``--target signet_ring``
fills in the recessed-design defaults so the user doesn't have to know
which knobs flip.

Each preset is a YAML file in ``profiles/targets/`` with shape::

    name: signet_ring
    display_name: Signet ring (recessed)
    print_width_mm: 30.0
    print_height_mm: 25.0
    polarity_invert: true
    heightmap:
      input_clahe: true
      subject_mask_enabled: true
    notes: >-
      ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


__all__ = [
    "TargetPreset",
    "list_target_presets",
    "load_target_preset",
    "DEFAULT_TARGETS_DIR",
]


# Repo-relative directory where the canonical target YAMLs live.
DEFAULT_TARGETS_DIR: Path = Path(__file__).resolve().parents[1] / "profiles" / "targets"


@dataclass(frozen=True)
class TargetPreset:
    """Loaded target preset. ``heightmap_overrides`` slots straight into
    ``merge_profile_settings(..., overrides=...)``."""

    name: str
    display_name: str
    print_width_mm: float
    print_height_mm: float
    polarity_invert: bool
    heightmap_overrides: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "print_width_mm": self.print_width_mm,
            "print_height_mm": self.print_height_mm,
            "polarity_invert": self.polarity_invert,
            "notes": self.notes,
        }


def _candidate_paths(name_or_path: str) -> List[Path]:
    raw = Path(name_or_path)
    if raw.exists():
        return [raw]
    stem = raw.stem if raw.suffix else name_or_path
    return [
        DEFAULT_TARGETS_DIR / f"{stem}.yaml",
        DEFAULT_TARGETS_DIR / f"{stem}.yml",
    ]


def load_target_preset(name_or_path: str) -> TargetPreset:
    for candidate in _candidate_paths(name_or_path):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
            return _from_dict(data, source=candidate)
    raise FileNotFoundError(f"target preset not found: {name_or_path!r}")


def list_target_presets() -> List[TargetPreset]:
    """Enumerate the shipped target presets, sorted by display name."""
    if not DEFAULT_TARGETS_DIR.exists():
        return []
    presets: List[TargetPreset] = []
    for path in sorted(DEFAULT_TARGETS_DIR.glob("*.y*ml")):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
            presets.append(_from_dict(data, source=path))
        except (OSError, yaml.YAMLError, ValueError):
            continue
    presets.sort(key=lambda p: p.display_name.lower())
    return presets


def _from_dict(data: Dict[str, Any], *, source: Optional[Path] = None) -> TargetPreset:
    try:
        return TargetPreset(
            name=str(data["name"]),
            display_name=str(data.get("display_name", data["name"])),
            print_width_mm=float(data["print_width_mm"]),
            print_height_mm=float(data["print_height_mm"]),
            polarity_invert=bool(data.get("polarity_invert", False)),
            heightmap_overrides=dict(data.get("heightmap") or {}),
            notes=str(data.get("notes", "")).strip(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        location = f" ({source})" if source is not None else ""
        raise ValueError(f"malformed target preset{location}: {exc}") from exc
