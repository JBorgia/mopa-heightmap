"""LightBurn ``.lbrn2`` color-card importer.

Parses LightBurn project files of the kind shipped under
``LightBurn Colour Card/`` (``Colour20W-M7.lbrn2`` … ``Colour100W-M7.lbrn2``)
and exposes them as :class:`MaterialProfile` objects suitable for the MOPA
color-pass planner.

Each card is a LightBurn project containing one ``<CutSetting type="Scan">``
per validated MOPA color, with the canonical machine parameters
(``maxPower``, ``speed``, ``frequency``, ``QPulseWidth``, ``interval``).
We treat those values as ground truth and never invent or interpolate them —
the planner lifts them verbatim into the exported ``.lbrn2``.

See ``IMPLEMENTATION_PLAN.md`` §5 for the role of this module in the pipeline.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional


__all__ = [
    "ColorEntry",
    "MaterialProfile",
    "load_lightburn_card",
    "load_all_cards",
    "DEFAULT_CARDS_DIR",
    "DEFAULT_PROFILE_NAME",
]


# Repo-relative location of the canonical color cards supplied by the user.
DEFAULT_CARDS_DIR = Path(__file__).resolve().parents[2] / "LightBurn Colour Card"

# The user's machine. Phase 11 decision: 60 W MOPA (M7) is the autoload default.
DEFAULT_PROFILE_NAME = "Colour60W-M7"


_INT_FIELDS = {
    "index", "minPower", "maxPower", "maxPower2", "speed", "frequency",
    "QPulseWidth", "priority", "tabCount", "tabCountMax", "bidir", "floodFill",
}
_FLOAT_FIELDS = {"interval"}
_STRING_FIELDS = {"name"}


@dataclass(frozen=True)
class ColorEntry:
    """One MOPA color, lifted verbatim from a LightBurn ``<CutSetting>``.

    All numeric fields are in LightBurn's own units (power %, mm/s, Hz, ns,
    mm) so they can be re-emitted into a ``.lbrn2`` without unit conversion.
    """

    index: int                       # LightBurn layer index (0..29 typical)
    name: str                        # human label, e.g. "C00"
    max_power: float                 # %
    speed: float                     # mm/s
    frequency: int                   # Hz
    q_pulse_width: int               # ns
    interval: float                  # mm (line spacing for raster)
    min_power: Optional[float] = None
    max_power_2: Optional[float] = None
    priority: int = 0
    flood_fill: bool = False
    bidir: Optional[bool] = None
    raw: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def cut_type(self) -> str:
        return "Scan"


@dataclass(frozen=True)
class MaterialProfile:
    """A parsed LightBurn color card.

    ``entries`` preserves the original LightBurn ordering (by ``index``).
    ``by_name`` and ``by_index`` give O(1) lookup. ``wattage`` is parsed
    from the filename when available (``Colour60W-M7.lbrn2`` -> 60).
    """

    name: str                                # canonical key, e.g. "Colour60W-M7"
    source_path: Path
    machine_label: str                       # raw DeviceName from XML or filename
    wattage: Optional[int]                   # tube wattage in W, parsed from name
    app_version: Optional[str]
    entries: List[ColorEntry]
    thumbnail_b64: Optional[str] = None      # whole-project preview, if present

    @property
    def by_name(self) -> Dict[str, ColorEntry]:
        return {e.name: e for e in self.entries}

    @property
    def by_index(self) -> Dict[int, ColorEntry]:
        return {e.index: e for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)


def _value_of(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    v = elem.attrib.get("Value")
    return v if v is not None else (elem.text or None)


def _parse_cut_setting(node: ET.Element) -> Optional[ColorEntry]:
    """Convert one ``<CutSetting>`` element into a :class:`ColorEntry`.

    Returns ``None`` for non-Scan cut settings (Cut, Image etc.) which the
    color-card schema is not expected to contain but which we politely ignore.
    """
    if node.attrib.get("type") not in (None, "Scan"):
        return None

    raw: Dict[str, str] = {}
    for child in node:
        v = _value_of(child)
        if v is not None:
            raw[child.tag] = v

    # Required fields. If any are missing the card is malformed; surface as
    # ValueError so the loader can attribute it to the right file.
    try:
        index = int(raw["index"])
        name = str(raw.get("name", f"C{index:02d}"))
        max_power = float(raw["maxPower"])
        speed = float(raw["speed"])
        frequency = int(float(raw["frequency"]))
        q_pulse_width = int(float(raw["QPulseWidth"]))
        interval = float(raw["interval"])
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"<CutSetting> missing required field {missing!r}") from None

    def _opt_float(key: str) -> Optional[float]:
        v = raw.get(key)
        return float(v) if v is not None else None

    def _opt_bool(key: str) -> Optional[bool]:
        v = raw.get(key)
        if v is None:
            return None
        return v not in ("0", "False", "false")

    return ColorEntry(
        index=index,
        name=name,
        max_power=max_power,
        speed=speed,
        frequency=frequency,
        q_pulse_width=q_pulse_width,
        interval=interval,
        min_power=_opt_float("minPower"),
        max_power_2=_opt_float("maxPower2"),
        priority=int(float(raw.get("priority", 0))),
        flood_fill=bool(_opt_bool("floodFill")) if "floodFill" in raw else False,
        bidir=_opt_bool("bidir"),
        raw=raw,
    )


_WATTAGE_RE = re.compile(r"(\d+)\s*W", re.IGNORECASE)


def _parse_wattage(stem: str) -> Optional[int]:
    m = _WATTAGE_RE.search(stem)
    return int(m.group(1)) if m else None


def load_lightburn_card(path: Path) -> MaterialProfile:
    """Parse a single LightBurn ``.lbrn2`` color card into a profile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LightBurn card not found: {path}")

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed LightBurn XML in {path}: {exc}") from exc

    root = tree.getroot()
    # Accept both the project (``LightBurnProject``) and Cut Library
    # (``CutSettings``) roots — both are read into the same ``MaterialProfile``.
    if root.tag not in ("LightBurnProject", "CutSettings"):
        raise ValueError(
            f"{path} is not a LightBurn project / Cut Library "
            f"(root tag {root.tag!r})"
        )

    entries: List[ColorEntry] = []
    seen_indices: set[int] = set()
    for node in root.findall("CutSetting"):
        try:
            entry = _parse_cut_setting(node)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from None
        if entry is None:
            continue
        if entry.index in seen_indices:
            raise ValueError(
                f"{path.name}: duplicate CutSetting index {entry.index}"
            )
        seen_indices.add(entry.index)
        entries.append(entry)

    if not entries:
        raise ValueError(f"{path.name}: no <CutSetting> entries found")

    entries.sort(key=lambda e: e.index)

    thumb = root.find("Thumbnail")
    thumb_b64 = thumb.attrib.get("Source") if thumb is not None else None

    return MaterialProfile(
        name=path.stem,
        source_path=path,
        machine_label=root.attrib.get("DeviceName", path.stem),
        wattage=_parse_wattage(path.stem),
        app_version=root.attrib.get("AppVersion"),
        entries=entries,
        thumbnail_b64=thumb_b64,
    )


def load_all_cards(
    cards_dir: Optional[Path] = None,
) -> Dict[str, MaterialProfile]:
    """Parse every ``*.lbrn2`` file in ``cards_dir`` (default: shipped cards).

    Returns a dict keyed by file stem (e.g. ``"Colour60W-M7"``). Files that
    fail to parse raise :class:`ValueError` immediately; we do not silently
    skip malformed cards because they'd map to wrong machine parameters.
    """
    base = Path(cards_dir) if cards_dir is not None else DEFAULT_CARDS_DIR
    if not base.exists():
        return {}
    profiles: Dict[str, MaterialProfile] = {}
    for path in sorted(base.glob("*.lbrn2")):
        profiles[path.stem] = load_lightburn_card(path)
    return profiles
