"""Emit a LightBurn ``.lbrn2`` project from a planned engraving stack.

Round-trip contract: parsing the output of :func:`write_lbrn` with
:func:`mopa.lightburn_cards.load_lightburn_card` MUST recover
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

import base64
import hashlib
import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
from xml.dom import minidom

import numpy as np
from PIL import Image

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
LBRN_DEFAULT_APP_VERSION: str = "1.7.00"

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

    For raster passes ``shape_type`` is ``"Bitmap"`` and ``source_path``
    points to the on-disk PNG (its bytes get embedded as base64 ``Data``
    so LightBurn can render the layer without a separate file load).
    For vector signature passes ``shape_type`` is ``"Path"`` and
    ``source_file`` is the SVG path.

    ``physical_width_mm`` / ``physical_height_mm`` set the bitmap's
    on-bed size; the XForm scale is computed from these and the
    bitmap's pixel dimensions so the printed image matches the
    requested physical size exactly.
    """

    cut_index: int
    shape_type: str = "Bitmap"
    source_file: Optional[str] = None
    source_path: Optional[Path] = None
    xform: tuple[float, float, float, float, float, float] = LBRN_IDENTITY_XFORM
    physical_width_mm: Optional[float] = None
    physical_height_mm: Optional[float] = None
    embed_data: bool = False
    # Bitmap clipping: set mask_id to the ShapeID of the Ellipse that clips it.
    mask_id: Optional[int] = None
    # Ellipse fields: Rx/Ry in mm, UsedByID = ShapeID of the Bitmap being clipped.
    rx: Optional[float] = None
    ry: Optional[float] = None
    used_by_id: Optional[int] = None
    # Workspace position of the shape centre (mm). When set, overrides the
    # default of placing the centre at (mm_w/2, mm_h/2) which lands the
    # image at the workspace origin. Pass (87.5, 87.5) to centre on a
    # 175 × 175 mm bed.
    center_x_mm: Optional[float] = None
    center_y_mm: Optional[float] = None


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


def _emit_cut_setting(
    parent: ET.Element,
    entry: ColorEntry,
    *,
    image_mode: bool = False,
    tool_mode: bool = False,
    image_pass_count: int = 256,
    image_negative: bool = False,
    image_dpi: int = 1270,
) -> None:
    """Append a ``<CutSetting type="Scan">`` (vector) or
    ``<CutSetting_Img type="Image">`` (raster, including 3D Slice) block.

    LightBurn uses two separate XML tag names for the two cut-setting
    flavours. Image-mode passes (the depth bitmap, photo-tonal, color
    anneal) need ``<CutSetting_Img>`` plus 3D-Slice-specific children
    (``ditherMode``, ``numPasses``, ``negative``, ``dpi``) so LightBurn
    opens them in the correct mode. Vector passes (signature text,
    cut-lines) use the original ``<CutSetting type="Scan">``.
    Tool-layer passes (reference circles, alignment marks) use
    ``<CutSetting type="Tool">``.
    """
    if image_mode:
        cs = ET.SubElement(parent, "CutSetting_Img", {"type": "Image"})
    elif tool_mode:
        cs = ET.SubElement(parent, "CutSetting", {"type": "Tool"})
    else:
        cs = ET.SubElement(parent, "CutSetting", {"type": entry.cut_type})

    if entry.raw:
        for tag, raw in entry.raw.items():
            child = ET.SubElement(cs, tag)
            child.set("Value", raw)
    else:
        # Fallback: synthesise from typed fields.
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

    if image_mode:
        # 3D-Slice children. ``ditherMode="3dslice"`` is the magic value
        # that makes LightBurn open the layer in 3D Slice mode without a
        # manual click. ``negative`` matches sculptok's bright_raised
        # polarity to LightBurn's "black is deep" convention (default 0).
        # If the cut already had a numPasses field via raw, we leave it.
        if not (entry.raw and "numPasses" in entry.raw):
            ET.SubElement(cs, "numPasses").set("Value", str(int(image_pass_count)))
        if not (entry.raw and "ditherMode" in entry.raw):
            ET.SubElement(cs, "ditherMode").set("Value", "3dslice")
        if not (entry.raw and "negative" in entry.raw):
            ET.SubElement(cs, "negative").set("Value", "1" if image_negative else "0")
        if not (entry.raw and "subname" in entry.raw):
            ET.SubElement(cs, "subname").set("Value", "3D Slice")
        if not (entry.raw and "dpi" in entry.raw):
            ET.SubElement(cs, "dpi").set("Value", str(int(image_dpi)))
        # Emit a cleanup SubLayer only when the card requests it.
        # An empty <SubLayer/> (no type/index) causes LightBurn to fail
        # parsing the entire CutSetting_Img, resulting in an empty Cuts panel.
        cleanup = (entry.raw or {}).get("cleanupPass")
        if cleanup == "1":
            max_pwr2 = (entry.raw or {}).get("maxPower2", "20")
            sl = ET.SubElement(cs, "SubLayer", {"type": "Scan", "index": "1"})
            ET.SubElement(sl, "maxPower").set("Value", max_pwr2)
            ET.SubElement(sl, "maxPower2").set("Value", max_pwr2)
            ET.SubElement(sl, "speed").set("Value", "2000")
            ET.SubElement(sl, "isCleanup").set("Value", "1")
            ET.SubElement(sl, "subname").set("Value", "Cleanup")


def _emit_shape(parent: ET.Element, ref: ShapeRef, *, shape_id: int = 0) -> None:
    """Emit a ``<Shape>`` element.

    For bitmap shapes with ``embed_data=True`` and ``source_path`` set,
    read the PNG, embed it as base64 ``Data`` (so LightBurn renders the
    image without a separate file), set ``W``/``H`` from the requested
    physical dimensions, and write the standard image-processing
    defaults (Gamma=1, Contrast/Brightness/Enhance*=0). The XForm scale
    is computed from physical-mm / pixel-count so the bitmap lands at
    the requested on-bed size.
    """
    attribs = {"Type": ref.shape_type, "CutIndex": str(ref.cut_index), "ShapeID": str(shape_id)}
    xform = ref.xform

    if ref.shape_type == "Bitmap" and ref.embed_data and ref.source_path is not None:
        png_path = Path(ref.source_path)
        png_bytes = png_path.read_bytes()
        with Image.open(io.BytesIO(png_bytes)) as img:
            px_w, px_h = img.size
        # Resolve physical size (default: 50 mm on the longest side).
        if ref.physical_width_mm and ref.physical_height_mm:
            mm_w = float(ref.physical_width_mm)
            mm_h = float(ref.physical_height_mm)
        else:
            longest = max(px_w, px_h)
            scale = 50.0 / float(longest)
            mm_w = float(ref.physical_width_mm or px_w * scale)
            mm_h = float(ref.physical_height_mm or px_h * scale)
        # LightBurn stores W/H as the image's NATURAL size in mm at its
        # 120-DPI reference scale. XForm a/d factors scale from that
        # natural size to the desired display size; (e, f) is the
        # workspace top-left corner. d is POSITIVE (no y-flip): MOPA
        # fiber/JCZFiber controllers use Y-downward coords so the image
        # renders naturally without an explicit vertical flip.
        _LBRN_NAT_DPI = 120.0
        w_natural = float(px_w) * 25.4 / _LBRN_NAT_DPI
        h_natural = float(px_h) * 25.4 / _LBRN_NAT_DPI
        sx = mm_w / w_natural
        sy = mm_h / h_natural
        # (e, f) positions the bitmap CENTER in the workspace. Placing it at
        # (mm_w/2, mm_h/2) lands the image at the midpoint of the print area,
        # matching the convention used by LightBurn when it exports projects
        # (e.g. the reference coin at 87.5, 87.5 on a 175 mm bed).
        cx = ref.center_x_mm if ref.center_x_mm is not None else mm_w / 2.0
        cy = ref.center_y_mm if ref.center_y_mm is not None else mm_h / 2.0
        xform = (sx, 0.0, 0.0, sy, cx, cy)
        if ref.mask_id is not None:
            attribs["MaskID"] = str(ref.mask_id)
        attribs.update({
            "W": _fmt_float(w_natural),
            "H": _fmt_float(h_natural),
            "Gamma": "1",
            "Contrast": "0",
            "Brightness": "0",
            "EnhanceAmount": "0",
            "EnhanceRadius": "0",
            "EnhanceDenoise": "0",
            "File": ref.source_file if ref.source_file is not None else png_path.name,
            "SourceHash": _source_hash(png_bytes),
            "Data": base64.b64encode(png_bytes).decode("ascii"),
        })

    elif ref.shape_type == "Ellipse":
        # Reference-circle / coin clipping mask. Rx/Ry are physical radii
        # in mm; XForm translation places the centre at (cx, cy) in the
        # workspace. UsedByID references the Bitmap this Ellipse clips.
        if ref.rx is not None:
            attribs["Rx"] = _fmt_float(ref.rx)
        if ref.ry is not None:
            attribs["Ry"] = _fmt_float(ref.ry)
        if ref.used_by_id is not None:
            attribs["UsedByID"] = str(ref.used_by_id)

    shape = ET.SubElement(parent, "Shape", attribs)
    xf = ET.SubElement(shape, "XForm")
    xf.text = " ".join(_fmt_float(v) for v in xform)


def _fmt_float(v: float) -> str:
    """Format a float the way LightBurn does — trim trailing zeros."""
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


def _source_hash(data: bytes) -> str:
    """Short integer hash for the SourceHash attribute (matches etcher's pattern)."""
    return str(int(hashlib.md5(data).hexdigest()[:6], 16) % 100)


# --------------------------------------------------- project boilerplate

def _emit_project_boilerplate(
    parent: ET.Element,
    *,
    thumbnail_b64: Optional[str] = None,
) -> None:
    """Emit the standard <Thumbnail>/<VariableText>/<UIPrefs> blocks.

    LightBurn's project parser drops layers when these are missing.
    Values match the ones the user's source 60W card ships with so the
    project loads with sensible defaults.
    """
    if thumbnail_b64:
        ET.SubElement(parent, "Thumbnail", {"Source": thumbnail_b64})

    vt = ET.SubElement(parent, "VariableText")
    for tag, val in (
        ("Start", "0"),
        ("End", "999"),
        ("Current", "0"),
        ("Increment", "1"),
        ("AutoAdvance", "0"),
    ):
        ET.SubElement(vt, tag, {"Value": val})

    ui = ET.SubElement(parent, "UIPrefs")
    for tag, val in (
        ("Optimize_ByLayer", "0"),
        ("Optimize_ByGroup", "-1"),
        ("Optimize_ByPriority", "1"),
        ("Optimize_WhichDirection", "0"),
        ("Optimize_InnerToOuter", "1"),
        ("Optimize_ByDirection", "0"),
        ("Optimize_ReduceTravel", "1"),
        ("Optimize_HideBacklash", "0"),
        ("Optimize_ReduceDirChanges", "0"),
        ("Optimize_ChooseCorners", "0"),
        ("Optimize_AllowReverse", "1"),
        ("Optimize_RemoveOverlaps", "0"),
        ("Optimize_OverlapDist", "0.025"),
        ("Optimize_OptimalEntryPoint", "0"),
        ("Optimize_CutSelected", "0"),
    ):
        ET.SubElement(ui, tag, {"Value": val})


def make_thumbnail_b64(heightmap: np.ndarray, max_side: int = 256) -> str:
    """Render a small base64 PNG thumbnail of ``heightmap`` for the project header."""
    arr = np.clip(heightmap.astype(np.float32, copy=False), 0.0, 1.0)
    img = Image.fromarray(
        (arr * 255.0 + 0.5).astype(np.uint8), mode="L",
    ).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------- builder

def build_lbrn_tree(
    *,
    entries: Sequence[ColorEntry],
    shapes: Sequence[ShapeRef],
    app_version: str = LBRN_DEFAULT_APP_VERSION,
    material_height: float = LBRN_DEFAULT_MATERIAL_HEIGHT,
    mirror_x: bool = False,
    mirror_y: bool = False,
    thumbnail_b64: Optional[str] = None,
    image_pass_count: int = 256,
    image_negative: bool = False,
    image_dpi: int = 1270,
    notes: Optional[str] = None,
) -> ET.ElementTree:
    """Construct the in-memory ``ElementTree`` for a LightBurn project.

    ``thumbnail_b64`` should be a base64-encoded PNG (e.g. produced by
    :func:`make_thumbnail_b64`). When omitted, the ``<Thumbnail>`` block
    is skipped — but LightBurn loads layers more reliably with one
    present, so callers should normally supply it.
    """
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
            "MaterialHeight": _fmt_float(material_height),
            "MirrorX": "True" if mirror_x else LBRN_DEFAULT_MIRROR_X,
            "MirrorY": "True" if mirror_y else LBRN_DEFAULT_MIRROR_Y,
        },
    )
    _emit_project_boilerplate(root, thumbnail_b64=thumbnail_b64)
    # An entry is emitted in image-mode when any shape attached to it is a
    # Bitmap (the .lbrn2's depth pass and color/photo-tonal passes). Vector
    # shapes (Path / Text / future signature) keep the Scan-style emit.
    # Ellipse shapes signal a Tool layer (index 30 in LightBurn) — emitted
    # as <CutSetting type="Tool"> rather than Scan or Image.
    image_indices = {s.cut_index for s in shapes if s.shape_type == "Bitmap"}
    tool_indices = {s.cut_index for s in shapes if s.shape_type == "Ellipse"}
    for entry in sorted(entries, key=lambda e: e.index):
        _emit_cut_setting(
            root, entry,
            image_mode=entry.index in image_indices,
            tool_mode=entry.index in tool_indices,
            image_pass_count=image_pass_count,
            image_negative=image_negative,
            image_dpi=image_dpi,
        )
    for i, shape in enumerate(shapes):
        _emit_shape(root, shape, shape_id=i)
    # Notes must come LAST — placing it before CutSetting_Img causes LightBurn
    # to silently drop all layers (its parser stops at the multi-line attribute).
    # Keep content ASCII-safe: no literal newlines or multi-byte × in attributes.
    if notes:
        safe_notes = notes.replace("\n", " | ").replace("×", "x")
        ET.SubElement(root, "Notes", {"ShowOnLoad": "0", "Notes": safe_notes})
    else:
        ET.SubElement(root, "Notes", {"ShowOnLoad": "0", "Notes": ""})
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
    print_width_mm: Optional[float] = None,
    print_height_mm: Optional[float] = None,
    thumbnail_b64: Optional[str] = None,
    image_pass_count: int = 256,
    image_negative: bool = False,
    image_dpi: int = 1270,
) -> Path:
    """Write a ``.lbrn2`` file to ``output_path`` from a :class:`PassPlan`.

    ``pass_pngs`` maps ``EngravingPass.id`` to the on-disk PNG to embed as
    that pass's bitmap shape. The PNG bytes are base64-encoded into the
    Shape's ``Data`` attribute so LightBurn renders the bitmap without a
    separate file load. Missing entries emit the shape as a Path
    placeholder (LightBurn loads it empty for vector signature passes).

    ``print_width_mm`` / ``print_height_mm`` set every Bitmap shape's
    on-bed physical size. When omitted, defaults to 50 mm on the longest
    side, preserving aspect ratio.

    ``thumbnail_b64`` should be a base64-encoded PNG preview (use
    :func:`make_thumbnail_b64`). LightBurn loads layers more reliably
    with one present.

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
            source_path=Path(png) if png is not None else None,
            embed_data=png is not None,
            physical_width_mm=print_width_mm,
            physical_height_mm=print_height_mm,
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
        thumbnail_b64=thumbnail_b64,
        image_pass_count=image_pass_count,
        image_negative=image_negative,
        image_dpi=image_dpi,
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
