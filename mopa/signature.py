"""Render a sub-mm relief signature into a corner of the heightmap.

The signature pass slot in :mod:`stages` exists; this module produces
the actual mask the user wants their machine to engrave for that slot.
By default we render a small text label (artist initials, year, machine
ID, …) in a chosen corner; advanced users can supply a pre-rendered PNG
mask instead.

Sizing:
    * ``corner`` picks one of "tl" | "tr" | "bl" | "br".
    * ``height_fraction`` is the text height as a fraction of the
      heightmap's shorter side. 0.04 (default) ≈ 0.4 mm on a 10 mm
      engraving — about right for a maker's mark on a coin / medallion.
    * ``margin_fraction`` is the inset from the chosen corner, same
      units. 0.03 keeps the mark visually inside the engraved area.

Output:
    Returns a float32 ``(H, W)`` mask in ``[0, 1]`` with the text region
    set to 1.0 and everywhere else 0.0. The pass planner consumes this
    via the ``masks`` parameter of :func:`stages.plan_passes`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


__all__ = [
    "render_text_signature_mask",
    "DEFAULT_HEIGHT_FRACTION",
    "DEFAULT_MARGIN_FRACTION",
    "DEFAULT_CORNER",
    "VALID_CORNERS",
]


DEFAULT_HEIGHT_FRACTION: float = 0.04
DEFAULT_MARGIN_FRACTION: float = 0.03
DEFAULT_CORNER: str = "br"
VALID_CORNERS: Tuple[str, ...] = ("tl", "tr", "bl", "br")


def _load_font(size: int) -> ImageFont.ImageFont:
    """Try to load a real font for crisp rendering; fall back to PIL bitmap."""
    for candidate in ("arialbd.ttf", "arial.ttf", "Verdana.ttf",
                      "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _corner_origin(
    canvas_w: int,
    canvas_h: int,
    text_w: int,
    text_h: int,
    margin_px: int,
    corner: str,
) -> Tuple[int, int]:
    if corner == "tl":
        return (margin_px, margin_px)
    if corner == "tr":
        return (canvas_w - text_w - margin_px, margin_px)
    if corner == "bl":
        return (margin_px, canvas_h - text_h - margin_px)
    if corner == "br":
        return (canvas_w - text_w - margin_px, canvas_h - text_h - margin_px)
    raise ValueError(f"corner must be one of {VALID_CORNERS}; got {corner!r}")


def render_text_signature_mask(
    shape: Tuple[int, int],
    text: str,
    *,
    corner: str = DEFAULT_CORNER,
    height_fraction: float = DEFAULT_HEIGHT_FRACTION,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    custom_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render ``text`` as a binary mask at ``shape``.

    Parameters
    ----------
    shape
        ``(H, W)`` of the output mask, matching the heightmap size.
    text
        Label to render. Empty string returns an all-zero mask.
    corner
        Which corner to anchor to. One of ``"tl" | "tr" | "bl" | "br"``.
    height_fraction
        Text height as a fraction of ``min(H, W)``. Default 4 %.
    margin_fraction
        Inset from the chosen corner, same fraction-of-shorter-side units.
    custom_mask
        Optional pre-rendered mask of any shape. If provided, the mask
        is bilinearly resized to ``shape`` and returned in place of the
        text rendering — lets users supply a logo / vector sigil.
    """
    if corner not in VALID_CORNERS:
        raise ValueError(f"corner must be one of {VALID_CORNERS}; got {corner!r}")
    H, W = int(shape[0]), int(shape[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"shape must be positive; got {shape}")
    out = np.zeros((H, W), dtype=np.float32)

    if custom_mask is not None:
        cm = np.asarray(custom_mask, dtype=np.float32)
        if cm.ndim != 2:
            raise ValueError(f"custom_mask must be 2-D; got {cm.shape}")
        if cm.shape != (H, W):
            cm = np.asarray(
                Image.fromarray(np.clip(cm, 0.0, 1.0), mode="F").resize(
                    (W, H), Image.BILINEAR,
                ),
                dtype=np.float32,
            )
        return np.clip(cm, 0.0, 1.0)

    if not text:
        return out

    short = min(H, W)
    font_px = max(8, int(round(short * float(height_fraction))))
    margin_px = max(1, int(round(short * float(margin_fraction))))

    font = _load_font(font_px)
    # Render onto a transparent PIL image to measure + paint, then composite.
    canvas = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    x, y = _corner_origin(W, H, text_w, text_h, margin_px, corner)
    # Subtract the bbox offset so the *visual* top-left lands at (x, y).
    draw.text((x - bbox[0], y - bbox[1]), text, fill=255, font=font)
    out = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)
