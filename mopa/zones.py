"""Zone mask computation and heightmap compositing for multi-layer engraving.

Zone vocabulary (in fire order):
  zone:field    — background plane; annealing parameters, fires first
  zone:border   — decorative band inside rim; intermediate parameters
  zone:rim      — outermost raised ring; low-power, fires 2nd/3rd
  zone:device   — central subject; ablation parameters, fires last
  zone:exergue  — bottom text strip; same priority as device

Internal zone separation is always raster-baked into per-pass PNGs.
LightBurn's MaskID can only clip a bitmap's outer boundary (handled
by the coin Ellipse), not sub-regions within the bitmap.

IMPORTANT: Never frequency-decompose the sculptok heightmap across zone
layers. Each zone carries its full depth budget independently — overlapping
pixels from the same heightmap at different frequency bands compound depth
3-5x across layers, burning through material.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .lightburn_cards import ColorEntry


__all__ = [
    "VALID_SHAPES",
    "zone_masks_from_geometry",
    "zone_hm_for_pass",
    "entries_from_zone_params",
]

# Accepted shape values for the profile `shape:` field.
VALID_SHAPES = frozenset({
    "circle",
    "rectangle",
    "hexagon",
    "triangle",
    "donut",
    "shield",
    "path",
})

# Base Gaussian sigma (px) applied at every zone boundary before sigma_scale.
_BASE_SIGMA_PX: float = 1.5

# LightBurn color-slot assignments for zone passes.
# device reuses C01 (same slot as form) so zones replace the form pass seamlessly.
ZONE_INDEX: Dict[str, int] = {
    "zone:field":    2,   # C02
    "zone:border":   3,   # C03
    "zone:rim":      4,   # C04
    "zone:device":   1,   # C01 — replaces form pass
    "zone:exergue":  5,   # C05
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _radial_distance(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Return float32 (H, W) pixel distances from (cx, cy)."""
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    return np.sqrt(xs * xs + ys * ys)


def _gaussian_feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a float32 mask, clamp to [0, 1]."""
    if sigma <= 0:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter
        return np.clip(
            gaussian_filter(mask.astype(np.float32), sigma=sigma),
            0.0, 1.0,
        )
    except ImportError:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _polygon_vertices(shape: str, cx: float, cy: float, r: float) -> np.ndarray:
    """Clockwise (N, 2) vertex array for a regular polygon."""
    if shape == "hexagon":
        n, start = 6, math.pi / 2
    elif shape == "triangle":
        n, start = 3, math.pi / 2
    else:
        raise ValueError(f"Unknown polygon shape: {shape!r}")
    angles = [start + 2 * math.pi * i / n for i in range(n)]
    return np.array(
        [[cx + r * math.cos(a), cy + r * math.sin(a)] for a in angles],
        dtype=np.float32,
    )


def _rasterise_polygon(verts: np.ndarray, h: int, w: int) -> np.ndarray:
    """Scanline-fill a polygon into a float32 (H, W) mask."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon(
            [(float(x), float(y)) for x, y in verts], fill=255
        )
        return np.asarray(img, dtype=np.float32) / 255.0
    except ImportError:
        # Fallback: bounding-circle approximation
        cx = float(verts[:, 0].mean())
        cy = float(verts[:, 1].mean())
        r = float(np.min(np.hypot(verts[:, 0] - cx, verts[:, 1] - cy)))
        return (_radial_distance(h, w, cx, cy) <= r).astype(np.float32)


def _polygon_inset(verts: np.ndarray, inset_px: float, h: int, w: int) -> np.ndarray:
    """Binary mask of the polygon shrunk inward by inset_px pixels."""
    if inset_px <= 0:
        return _rasterise_polygon(verts, h, w)
    try:
        from shapely.geometry import Polygon as _Poly
        shrunk = _Poly(verts.tolist()).buffer(-inset_px)
        if shrunk.is_empty:
            return np.zeros((h, w), dtype=np.float32)
        coords = np.array(list(shrunk.exterior.coords), dtype=np.float32)
        return _rasterise_polygon(coords, h, w)
    except ImportError:
        # Shapely absent — erode the raster mask
        outer = _rasterise_polygon(verts, h, w)
        try:
            from scipy.ndimage import binary_erosion
            size = max(1, int(inset_px * 2 + 1))
            struct = np.ones((size, size), dtype=bool)
            return binary_erosion(outer > 0.5, structure=struct).astype(np.float32)
        except ImportError:
            return outer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def zone_masks_from_geometry(
    h: int,
    w: int,
    shape: str,
    *,
    print_w_mm: float,
    print_h_mm: float,
    px_per_mm: float,
    border_width_mm: float = 1.5,
    rim_width_mm: float = 0.5,
    exergue_height_fraction: float = 0.18,
    hole_radius_mm: Optional[float] = None,
    sigma_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute float32 (H, W) zone masks for a blank of the given geometry.

    Returns a dict with keys: ``outer``, ``field``, ``border``, ``rim``,
    ``exergue``. The ``device`` (subject) mask is user-provided via the
    ``/mask`` route and is NOT computed here.

    Parameters
    ----------
    h, w
        Heightmap pixel dimensions.
    shape
        Blank shape: ``circle`` | ``rectangle`` | ``hexagon`` | ``triangle``
        | ``donut`` | ``shield`` | ``path``. Unrecognised values fall back to
        rectangle.
    print_w_mm, print_h_mm
        Physical bounding box of the blank in mm.
    px_per_mm
        Pixels per mm at the current heightmap resolution.
    border_width_mm
        Radial/inset width of the border zone in mm.
    rim_width_mm
        Radial/inset width of the rim zone in mm.
    exergue_height_fraction
        Fraction of bounding-box height allocated to the bottom text strip.
        Shape-agnostic — applied as ``y > (1 - f) * h``.
    hole_radius_mm
        Inner hole radius for donut/annular blanks. ``None`` = solid.
    sigma_scale
        Multiplies the base feathering sigma (1.5 px). Use profile field
        ``zone_boundary_sigma_scale``.
    """
    cx, cy = w / 2.0, h / 2.0
    sigma = _BASE_SIGMA_PX * sigma_scale
    border_px = border_width_mm * px_per_mm
    rim_px = rim_width_mm * px_per_mm

    # ------------------------------------------------------------------
    # Build outer, rim, and border masks per shape
    # ------------------------------------------------------------------
    if shape in ("circle", "donut"):
        outer_r = min(print_w_mm, print_h_mm) / 2.0 * px_per_mm
        dist = _radial_distance(h, w, cx, cy)
        outer_mask = (dist <= outer_r).astype(np.float32)
        rim_mask   = ((dist > outer_r - rim_px) & (dist <= outer_r)).astype(np.float32)
        inner_r = outer_r - rim_px
        border_mask = (
            (dist > inner_r - border_px) & (dist <= inner_r)
        ).astype(np.float32)

        if shape == "donut" and hole_radius_mm is not None and hole_radius_mm > 0:
            hole_r = hole_radius_mm * px_per_mm
            hole = dist <= hole_r
            outer_mask[hole] = 0.0
            rim_mask[hole]   = 0.0
            border_mask[hole] = 0.0
            # Additional rim band at the inner hole edge
            inner_rim = (dist >= hole_r) & (dist < hole_r + rim_px)
            rim_mask = np.clip(rim_mask + inner_rim, 0.0, 1.0)

    elif shape == "rectangle":
        rx = print_w_mm / 2.0 * px_per_mm
        ry = print_h_mm / 2.0 * px_per_mm
        xs = np.abs(np.arange(w, dtype=np.float32) - cx)
        ys = np.abs((np.arange(h, dtype=np.float32) - cy)[:, None])
        in_outer = (xs <= rx) & (ys <= ry)
        outer_mask = in_outer.astype(np.float32)
        in_rim = in_outer & ((xs > rx - rim_px) | (ys > ry - rim_px))
        rim_mask = in_rim.astype(np.float32)
        in_border = in_outer & ~in_rim & (
            (xs > rx - rim_px - border_px) | (ys > ry - rim_px - border_px)
        )
        border_mask = in_border.astype(np.float32)

    elif shape in ("hexagon", "triangle"):
        outer_r = min(print_w_mm, print_h_mm) / 2.0 * px_per_mm
        verts = _polygon_vertices(shape, cx, cy, outer_r)
        outer_mask  = _rasterise_polygon(verts, h, w)
        rim_inset   = _polygon_inset(verts, rim_px, h, w)
        rim_mask    = np.clip(outer_mask - rim_inset, 0.0, 1.0)
        border_inset = _polygon_inset(verts, rim_px + border_px, h, w)
        border_mask  = np.clip(rim_inset - border_inset, 0.0, 1.0)

    else:
        # shield / path / unknown — treat as rectangle
        rx = print_w_mm / 2.0 * px_per_mm
        ry = print_h_mm / 2.0 * px_per_mm
        xs = np.abs(np.arange(w, dtype=np.float32) - cx)
        ys = np.abs((np.arange(h, dtype=np.float32) - cy)[:, None])
        in_outer = (xs <= rx) & (ys <= ry)
        outer_mask = in_outer.astype(np.float32)
        in_rim = in_outer & ((xs > rx - rim_px) | (ys > ry - rim_px))
        rim_mask = in_rim.astype(np.float32)
        in_border = in_outer & ~in_rim & (
            (xs > rx - rim_px - border_px) | (ys > ry - rim_px - border_px)
        )
        border_mask = in_border.astype(np.float32)

    # ------------------------------------------------------------------
    # Interior = outer minus rim minus border (shared by field + exergue)
    # ------------------------------------------------------------------
    interior_mask = np.clip(outer_mask - rim_mask - border_mask, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Exergue: bottom fraction of bounding-box height — shape-agnostic,
    # confined INSIDE the border/rim rings (a text strip never overprints
    # the decorative bands). Suppressed for donut.
    # ------------------------------------------------------------------
    exergue_y0 = int(h * (1.0 - exergue_height_fraction))
    exergue_mask = np.zeros((h, w), dtype=np.float32)
    if shape != "donut":
        exergue_mask[exergue_y0:, :] = interior_mask[exergue_y0:, :]

    # ------------------------------------------------------------------
    # Field = interior minus exergue — zones must be mutually exclusive
    # or the field and exergue passes double-engrave the bottom strip.
    # ------------------------------------------------------------------
    field_mask = np.clip(interior_mask - exergue_mask, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Feather all masks
    # ------------------------------------------------------------------
    return {
        "outer":    _gaussian_feather(outer_mask,   sigma),
        "field":    _gaussian_feather(field_mask,   sigma),
        "border":   _gaussian_feather(border_mask,  sigma),
        "rim":      _gaussian_feather(rim_mask,     sigma),
        "exergue":  _gaussian_feather(exergue_mask, sigma),
    }


def zone_hm_for_pass(
    heightmap: np.ndarray,
    zone_mask: np.ndarray,
) -> np.ndarray:
    """Composite a heightmap for one zone pass.

    Pixels outside the zone (zone_mask ≈ 0) are raised to 1.0 (LightBurn
    surface — no engraving). Pixels inside (zone_mask ≈ 1) retain the
    original sculptok depth.

    After compositing a bilateral filter suppresses dither-artifact halos
    at zone seams without smearing the transition boundary.
    """
    result = np.clip(
        1.0 - (1.0 - heightmap) * zone_mask,
        0.0, 1.0,
    ).astype(np.float32)

    # Bilateral post-filter on float32 (CV_32F) — avoids the 8-bit
    # precision loss of the uint8 round-trip.  sigmaColor ≈ 0.10 is
    # equivalent to 25/255 in the [0, 1] float domain.
    try:
        import cv2
        result = cv2.bilateralFilter(result, d=9, sigmaColor=0.10, sigmaSpace=10)
    except ImportError:
        pass

    return result


def mask_from_heightmap(
    heightmap: np.ndarray,
    *,
    bg_tol: float = 0.03,
    feather_px: int = 3,
) -> Optional[np.ndarray]:
    """Derive a subject/device mask from the heightmap itself.

    Sculptok reliefs sit on a flat background (the border-median level);
    any pixel that departs from it is subject. This beats photo-inference
    masks (rembg/birefnet) for drawings and engravings where the "subject"
    is line art the photo models segment poorly.

    Returns a float32 (H, W) mask in [0, 1] (1 = subject), or ``None``
    when the derivation is degenerate (background not flat / coverage
    outside 1–95 %) so callers can fall back to a photo mask.
    """
    hm = heightmap.astype(np.float32, copy=False)
    border = np.concatenate([hm[0, :], hm[-1, :], hm[:, 0], hm[:, -1]])
    bg = float(np.median(border))
    # Background must actually be flat — otherwise this heightmap wasn't
    # background-flattened and thresholding would slice through relief.
    if float(np.abs(border - bg).mean()) > bg_tol * 2:
        return None

    mask = (np.abs(hm - bg) > bg_tol).astype(np.float32)
    coverage = float(mask.mean())
    if coverage < 0.01 or coverage > 0.95:
        return None

    try:
        import cv2
        # Close pinholes inside the subject, drop speckle outside, then
        # feather the boundary so the device/field seam doesn't ring.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if feather_px > 0:
            k = feather_px * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
    except ImportError:
        pass
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def entries_from_zone_params(
    zone_params: Dict[str, Any],
    starting: Dict[str, Any],
    *,
    cleanup_every_passes: Optional[int] = None,
) -> Dict[str, ColorEntry]:
    """Build a ColorEntry for each zone defined in zone_params.

    Falls back to ``starting`` (lightburn_starting_point) for the device
    zone so that zone:device matches the main 3D-sculpt parameters.

    Returns a dict ``{zone_kind: ColorEntry}``.
    """
    # Zones supported and their default fallbacks (from starting_point)
    zone_spec: List[tuple[str, Dict[str, Any]]] = []
    for zone_kind in ("zone:field", "zone:border", "zone:rim", "zone:device", "zone:exergue"):
        short = zone_kind.split(":")[1]   # "field", "border", ...
        params = dict(zone_params.get(short) or {})
        if not params and zone_kind == "zone:device":
            params = dict(starting)       # device defaults to starting_point
        if not params:
            continue                      # zone not configured — skip
        zone_spec.append((zone_kind, params))

    entries: Dict[str, ColorEntry] = {}
    for zone_kind, params in zone_spec:
        idx = ZONE_INDEX[zone_kind]
        short = zone_kind.split(":")[1].capitalize()

        speed    = float(params.get("speed_mm_s",    starting.get("speed_mm_s",    1000)))
        power    = float(params.get("power_percent",  starting.get("power_percent",  65)))
        freq_khz = float(params.get("frequency_khz", starting.get("frequency_khz",  80)))
        pulse_ns = int(  params.get("pulse_width_ns", starting.get("pulse_width_ns", 180)))
        interval = float(params.get("line_interval_mm", starting.get("line_interval_mm", 0.02)))
        passes   = int(  params.get("passes",          starting.get("passes",          256)))

        raw: Dict[str, Any] = {
            "index":      str(idx),
            "name":       f"Zone{short}",
            "maxPower":   str(int(power)),
            "speed":      str(int(speed)),
            "frequency":  str(int(freq_khz * 1000)),
            "QPulseWidth": str(pulse_ns),
            "interval":   str(interval),
            "numPasses":  str(passes),
            "ditherMode": "3dslice",
            "negative":   "0",
            "subname":    f"Zone {short}",
            "dpi":        "1270",
        }
        if cleanup_every_passes and zone_kind == "zone:device":
            raw["cleanupPass"] = "1"

        entries[zone_kind] = ColorEntry(
            index=idx,
            name=f"Zone{short}",
            max_power=power,
            speed=speed,
            frequency=int(freq_khz * 1000),
            q_pulse_width=pulse_ns,
            interval=interval,
            raw=raw,
        )

    return entries
