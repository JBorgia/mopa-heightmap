"""Emit a LightBurn ``.lbrn2`` project from a planned engraving stack.

Round-trip contract: parsing the output of :func:`write_lbrn` with
:func:`zoedepth.laser.lightburn_cards.load_lightburn_card` MUST recover
``ColorEntry`` rows whose ``raw`` payload matches the original card
verbatim. We never rewrite, scale, or re-derive the machine parameters.

The writer outputs the schema we discovered while inspecting the user's
``Colour60W-M7.lbrn2`` card:

.. code-block:: xml

    <LightBurnProject AppVersion="..." FormatVersion="1"
                      MaterialHeight="..." MirrorX="False" MirrorY="False">
      <CutSetting type="Scan">
        <index Value="0"/>
        <name Value="C00"/>
        <maxPower Value="..."/>
        ...verbatim ColorEntry.raw values...
      </CutSetting>
      ...
      <Shape Type="Bitmap" CutIndex="0" SourceFile="form.png">
        <XForm>1 0 0 1 0 0</XForm>
      </Shape>
      ...
    </LightBurnProject>

PNGs are referenced by relative path so the project file remains portable
as long as the user keeps the bundle directory together.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
from xml.dom import minidom

from .lightburn_cards import ColorEntry, MaterialProfile
from .stages import EngravingPass, PassPlan


__all__ = [
    "write_lbrn",
    "write_clb",
    "build_lbrn_tree",
    "build_clb_tree",
    "ShapeRef",
    "LBRN_FORMAT_VERSION",
    "LBRN_DEFAULT_APP_VERSION",
    "LBRN_DEFAULT_MIRROR_X",
    "LBRN_DEFAULT_MIRROR_Y",
    "LBRN_DEFAULT_MATERIAL_HEIGHT",
    "LBRN_IDENTITY_XFORM",
    "CLB_ROOT_TAG",
]


# Root tag for the LightBurn Cut Library (.clb) format. A .clb file is a
# bare list of <CutSetting> blocks under <CutSettings> — same XML schema
# we already lift verbatim from a material card, no shapes, no project
# wrapper. LightBurn imports it via Settings → Library → Import.
CLB_ROOT_TAG: str = "CutSettings"


# ----------------------------------------------------------- format constants

# The .lbrn2 schema version; LightBurn 1.0+ uses "1" here.
LBRN_FORMAT_VERSION: str = "1"

# Fallback AppVersion stamped into freshly-authored projects when no source
# card AppVersion is available. Chosen to match the user's supplied cards so
# downstream tooling sees a familiar version string.
LBRN_DEFAULT_APP_VERSION: str = "1.2.04"

# Whole-project mirror flags. Defaults match the supplied 60W card.
LBRN_DEFAULT_MIRROR_X: str = "False"
LBRN_DEFAULT_MIRROR_Y: str = "False"

# Default material height in mm. 0 means "use machine bed default".
LBRN_DEFAULT_MATERIAL_HEIGHT: float = 0.0

# 2-D affine identity used for shapes that don't need to be transformed.
# LightBurn stores XForm as six space-separated numbers: a b c d e f
# (corresponding to the 2x3 matrix ``[[a, c, e], [b, d, f]]``).
LBRN_IDENTITY_XFORM: tuple[float, float, float, float, float, float] = (
    1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
)


# --------------------------------------------------------------- shape refs

@dataclass(frozen=True)
class ShapeRef:
    """One shape attached to a LightBurn cut layer.

    For raster passes ``shape_type`` is ``"Bitmap"`` and ``source_file`` is
    the relative path to the per-pass PNG. For vector signature passes
    ``shape_type`` is ``"Path"`` and ``source_file`` is the SVG path.
    """

    cut_index: int
    shape_type: str = "Bitmap"
    source_file: Optional[str] = None
    xform: tuple[float, float, float, float, float, float] = LBRN_IDENTITY_XFORM


# ---------------------------------------------------------- value helpers

def _set_value(parent: ET.Element, tag: str, value) -> None:
    """Append a ``<tag Value="..."/>`` child to ``parent``."""
    if value is None:
        return
    child = ET.SubElement(parent, tag)
    if isinstance(value, bool):
        child.set("Value", "1" if value else "0")
    elif isinstance(value, float):
        # Match LightBurn's float formatting: trim trailing zeroes but keep a
        # visible decimal so re-parse stays float (not int).
        text = repr(value)
        child.set("Value", text)
    else:
        child.set("Value", str(value))


def _emit_cut_setting(parent: ET.Element, entry: ColorEntry) -> None:
    """Append a ``<CutSetting type="Scan">`` block built verbatim from ``entry``.

    We prefer the raw payload captured at parse time so any fields we don't
    explicitly model (tabCount, priority2, etc.) round-trip unchanged.
    """
    cs = ET.SubElement(parent, "CutSetting", {"type": entry.cut_type})
    if entry.raw:
        for tag, raw in entry.raw.items():
            child = ET.SubElement(cs, tag)
            child.set("Value", raw)
        return
    # Fallback: synthesise from typed fields when raw is unavailable
    # (programmatically constructed entries).
    _set_value(cs, "index", entry.index)
    _set_value(cs, "name", entry.name)
    _set_value(cs, "maxPower", entry.max_power)
    _set_value(cs, "speed", entry.speed)
    _set_value(cs, "frequency", entry.frequency)
    _set_value(cs, "QPulseWidth", entry.q_pulse_width)
    _set_value(cs, "interval", entry.interval)
    _set_value(cs, "minPower", entry.min_power)
    _set_value(cs, "maxPower2", entry.max_power_2)
    _set_value(cs, "priority", entry.priority)
    _set_value(cs, "floodFill", entry.flood_fill)
    if entry.bidir is not None:
        _set_value(cs, "bidir", entry.bidir)


def _emit_shape(parent: ET.Element, ref: ShapeRef) -> None:
    attribs = {"Type": ref.shape_type, "CutIndex": str(ref.cut_index)}
    if ref.source_file is not None:
        attribs["SourceFile"] = ref.source_file
    shape = ET.SubElement(parent, "Shape", attribs)
    xf = ET.SubElement(shape, "XForm")
    xf.text = " ".join(repr(v) for v in ref.xform)


# --------------------------------------------------------------- builder

def build_lbrn_tree(
    *,
    entries: Sequence[ColorEntry],
    shapes: Sequence[ShapeRef],
    app_version: str = LBRN_DEFAULT_APP_VERSION,
    material_height: float = LBRN_DEFAULT_MATERIAL_HEIGHT,
    mirror_x: bool = False,
    mirror_y: bool = False,
) -> ET.ElementTree:
    """Construct the in-memory ``ElementTree`` for a LightBurn project."""
    seen: set[int] = set()
    for e in entries:
        if e.index in seen:
            raise ValueError(f"Duplicate CutSetting index in entries: {e.index}")
        seen.add(e.index)

    valid_indices = {e.index for e in entries}
    for s in shapes:
        if s.cut_index not in valid_indices:
            raise ValueError(
                f"Shape references CutIndex {s.cut_index} which has no CutSetting"
            )

    root = ET.Element(
        "LightBurnProject",
        {
            "AppVersion": app_version,
            "FormatVersion": LBRN_FORMAT_VERSION,
            "MaterialHeight": repr(float(material_height)),
            "MirrorX": "True" if mirror_x else LBRN_DEFAULT_MIRROR_X,
            "MirrorY": "True" if mirror_y else LBRN_DEFAULT_MIRROR_Y,
        },
    )
    for entry in sorted(entries, key=lambda e: e.index):
        _emit_cut_setting(root, entry)
    for shape in shapes:
        _emit_shape(root, shape)
    return ET.ElementTree(root)


# --------------------------------------------------------------- public API

def write_lbrn(
    output_path: Path,
    plan: PassPlan,
    *,
    pass_pngs: Mapping[str, Path] | None = None,
    app_version: Optional[str] = None,
    material_height: Optional[float] = None,
    mirror_x: Optional[bool] = None,
    mirror_y: Optional[bool] = None,
    extra_entries: Iterable[ColorEntry] = (),
) -> Path:
    """Write a ``.lbrn2`` file to ``output_path`` from a :class:`PassPlan`.

    ``pass_pngs`` maps ``EngravingPass.id`` to the on-disk PNG to embed as
    that pass's bitmap shape. Missing entries omit the ``SourceFile``
    attribute (LightBurn will load the layer empty, which is what we want
    for vector / signature passes).

    ``extra_entries`` lets callers append unused-but-known cut settings
    (e.g. the rest of the material card) so users can re-enable them inside
    LightBurn without re-importing the source card.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = plan.profile
    used: dict[int, ColorEntry] = {}
    shapes: list[ShapeRef] = []
    for ep in plan.passes:
        used[ep.cut_setting.index] = ep.cut_setting
        png = (pass_pngs or {}).get(ep.id)
        rel = (
            os.path.relpath(
                str(Path(png).resolve()),
                start=str(output_path.parent.resolve()),
            )
            if png is not None
            else None
        )
        shapes.append(ShapeRef(
            cut_index=ep.cut_setting.index,
            shape_type="Bitmap" if png is not None else "Path",
            source_file=rel,
        ))
    for entry in extra_entries:
        used.setdefault(entry.index, entry)

    tree = build_lbrn_tree(
        entries=tuple(used.values()),
        shapes=tuple(shapes),
        app_version=app_version or profile.app_version or LBRN_DEFAULT_APP_VERSION,
        material_height=(
            material_height
            if material_height is not None
            else LBRN_DEFAULT_MATERIAL_HEIGHT
        ),
        mirror_x=False if mirror_x is None else bool(mirror_x),
        mirror_y=False if mirror_y is None else bool(mirror_y),
    )
    # Pretty-print so the file diffs cleanly when committed alongside outputs.
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(
        indent="  ", encoding="utf-8"
    )
    output_path.write_bytes(pretty)
    return output_path


# --------------------------------------------------------------- .clb writer

def build_clb_tree(entries: Sequence[ColorEntry]) -> ET.ElementTree:
    """Construct a LightBurn ``.clb`` Cut Library element tree.

    Output schema:

    .. code-block:: xml

        <CutSettings>
          <CutSetting type="Scan">...verbatim from each ColorEntry...</CutSetting>
          ...
        </CutSettings>

    The ``CutSetting`` blocks are emitted byte-identically to the source
    material card so the round-trip into the LightBurn Library carries
    every field upstream had — including ones we don't model.
    """
    seen: set[int] = set()
    for e in entries:
        if e.index in seen:
            raise ValueError(f"Duplicate CutSetting index in entries: {e.index}")
        seen.add(e.index)

    root = ET.Element(CLB_ROOT_TAG)
    for entry in sorted(entries, key=lambda e: e.index):
        _emit_cut_setting(root, entry)
    return ET.ElementTree(root)


def write_clb(
    output_path: Path,
    entries: Sequence[ColorEntry],
) -> Path:
    """Write a LightBurn Cut Library ``.clb`` file to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_clb_tree(entries)
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(
        indent="  ", encoding="utf-8"
    )
    output_path.write_bytes(pretty)
    return output_path
