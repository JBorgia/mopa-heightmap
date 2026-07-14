"""Post-sculptok zone overlay generators.

These run AFTER the sculptok heightmap is loaded and enhanced, compositing
geometry-aware patterns into the zone:rim and zone:border regions of the
heightmap.  The zone:field overlay (radial_lines, guilloche, etc.) uses the
existing backgrounds.generators functions, dispatched from apply_zone_overlays.

All generators return a float32 (H, W) array in [0, 1] where:
    1.0 = surface (white, no engraving)
    0.0 = deepest cut (black)

This matches LightBurn's black_is_deep convention so overlays can be
composited directly into the master heightmap with np.where / masking.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Rim: beaded pattern
# ---------------------------------------------------------------------------

def beaded_rim_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    bead_count: int = 72,
    bead_depth: float = 1.0,
) -> np.ndarray:
    """Hemispherical beads equally spaced around a circular rim.

    Beads sit at ``outer_r_px - rim_width_px/2`` (mid-rim).  Each bead is a
    spherical-cap paraboloid clipped to its own radius.  Gaps between beads
    are at 0.0 (maximum depth).

    Parameters
    ----------
    bead_depth:
        Peak value of each bead hemisphere.  1.0 = fully raised to surface.
    """
    center_r = outer_r_px - rim_width_px * 0.5
    # Bead radius = half the arc spacing, capped to ≤ rim_width/2.
    arc_spacing = 2.0 * math.pi * center_r / max(bead_count, 1)
    bead_r = min(arc_spacing * 0.46, rim_width_px * 0.46)

    out = np.zeros((h, w), dtype=np.float32)

    for i in range(bead_count):
        angle = 2.0 * math.pi * i / bead_count
        bx = cx + center_r * math.cos(angle)
        by = cy + center_r * math.sin(angle)

        # Bounding box — slightly larger than bead radius for clean edges.
        pad = bead_r * 1.1
        x0 = max(0, int(bx - pad))
        x1 = min(w, int(bx + pad) + 1)
        y0 = max(0, int(by - pad))
        y1 = min(h, int(by + pad) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = (xx - bx) ** 2 + (yy - by) ** 2
        r2 = bead_r ** 2
        # Paraboloid cap: val = 1 − (d/r)²   inside bead, 0 outside.
        cap = np.where(d2 < r2, (1.0 - d2 / r2) * bead_depth, 0.0).astype(np.float32)
        np.maximum(out[y0:y1, x0:x1], cap, out=out[y0:y1, x0:x1])

    return out


# ---------------------------------------------------------------------------
# Rim: reeded (milled coin edge)
# ---------------------------------------------------------------------------

def reeded_rim_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    reed_count: int = 120,
    reed_depth: float = 1.0,
) -> np.ndarray:
    """Radial reeding (milled coin edge) in the rim annulus.

    Evenly spaced radial ridges like the milled edge of a coin, but laid
    flat on the face rim.  Each reed has a rounded (sinusoidal) profile:
    crest at ridge centre, groove floor between reeds.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    theta = np.arctan2(ys, xs)

    # Rounded ridge profile: 1.0 at each reed crest, 0.0 in each groove.
    n = max(int(reed_count), 4)
    profile = 0.5 + 0.5 * np.cos(theta * n)
    # Sharpen slightly so crests read as distinct reeds, not a soft wave.
    profile = np.power(np.clip(profile, 0.0, 1.0), 0.75)

    return np.clip(profile * float(np.clip(reed_depth, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Rim: denticles (classic coin teeth)
# ---------------------------------------------------------------------------

def denticled_rim_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    element_count: int = 96,
    depth: float = 1.0,
) -> np.ndarray:
    """Denticles — the rectangular teeth ringing classic coin rims.

    Raised radial blocks with a ~55% duty cycle and soft shoulders.  The
    teeth grow from the outer edge inward and stop just short of the
    rim's inner boundary so each tooth reads as a discrete block.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r = np.sqrt(xs * xs + ys * ys)
    theta = np.arctan2(ys, xs)

    n = max(int(element_count), 8)
    frac = ((theta / (2.0 * math.pi) + 0.5) * n) % 1.0

    duty, edge = 0.55, 0.10
    up = np.clip(frac / edge, 0.0, 1.0)
    down = np.clip((duty - frac) / edge + 1.0, 0.0, 1.0)
    tooth = np.minimum(up, down)
    tooth = tooth * tooth * (3.0 - 2.0 * tooth)  # smoothstep shoulders

    # Radial taper: teeth start a little inside the rim's inner boundary.
    inner = outer_r_px - rim_width_px
    r_norm = np.clip((r - inner) / max(rim_width_px, 1.0), 0.0, 1.0)
    radial = np.clip((r_norm - 0.12) / 0.10, 0.0, 1.0)

    out = tooth * radial * float(np.clip(depth, 0.0, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Rim: rope cable
# ---------------------------------------------------------------------------

def rope_rim_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    element_count: int = 90,
    depth: float = 1.0,
) -> np.ndarray:
    """Twisted cable confined to the rim band — a rope rim.

    Same helical construction as the border rope but tuned for the much
    narrower rim: two strands, tighter twist, rounded Gaussian profile.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r = np.sqrt(xs * xs + ys * ys)
    theta = np.arctan2(ys, xs)

    inner = outer_r_px - rim_width_px
    r_norm = (r - inner) / max(rim_width_px, 1.0)
    in_band = (r >= inner) & (r <= outer_r_px)

    n = max(int(element_count), 8)
    sigma = 0.17
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(2):
        centre = 0.5 + 0.4 * np.sin(theta * n + math.pi * i)
        val = np.exp(-((r_norm - centre) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        np.maximum(out, val, out=out)

    out *= float(np.clip(depth, 0.0, 1.0))
    out[~in_band] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Rim: serrated (sharp triangular teeth)
# ---------------------------------------------------------------------------

def serrated_rim_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    element_count: int = 80,
    depth: float = 1.0,
) -> np.ndarray:
    """Serrations — sharp triangular ridges like a serrated coin edge.

    Triangle wave around the circumference, sharpened so the valleys are
    wider than the crests (each ridge reads as a distinct point).
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    theta = np.arctan2(ys, xs)

    n = max(int(element_count), 8)
    frac = ((theta / (2.0 * math.pi) + 0.5) * n) % 1.0
    tri = 1.0 - np.abs(2.0 * frac - 1.0)
    sharp = np.power(tri, 1.6)  # narrow the crest, widen the valley

    out = sharp * float(np.clip(depth, 0.0, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Border: rope twist
# ---------------------------------------------------------------------------

def rope_twist_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    border_width_px: float,
    strand_count: int = 2,
    twist_periods: int = 40,
) -> np.ndarray:
    """Helical rope/cable in the border annulus.

    Each strand follows a sinusoidal path through the border's radial
    extent (inner to outer edge), winding ``twist_periods`` times around
    the full circumference.  The Gaussian cross-section gives each strand
    a rounded, rope-like profile.

    Parameters
    ----------
    strand_count:
        Number of interleaved strands (2 = classic twisted pair).
    twist_periods:
        Full 360° helical cycles around the annulus.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r = np.sqrt(xs * xs + ys * ys)
    theta = np.arctan2(ys, xs)

    border_outer = outer_r_px - rim_width_px
    border_inner = border_outer - border_width_px
    border_span = max(border_outer - border_inner, 1.0)

    # Normalised radial position: 0 = inner edge of border, 1 = outer edge.
    r_norm = (r - border_inner) / border_span
    in_border = (r >= border_inner) & (r <= border_outer)

    # Strand width (sigma) as fraction of radial span.
    sigma = 0.18 / max(strand_count, 1)

    strand_max = np.zeros((h, w), dtype=np.float32)
    for i in range(strand_count):
        phase = 2.0 * math.pi * i / strand_count
        # Strand centreline oscillates radially as it winds angularly.
        centre = 0.5 + 0.45 * np.sin(theta * twist_periods + phase)
        dist = np.abs(r_norm - centre)
        # Gaussian profile — strand is bright at centreline, fades to 0.
        val = np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        np.maximum(strand_max, val, out=strand_max)

    out = np.clip(strand_max, 0.0, 1.0)
    out[~in_border] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Border: acanthus wave (botanical scroll approximation)
# ---------------------------------------------------------------------------

def acanthus_wave_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    border_width_px: float,
    leaf_count: int = 12,
) -> np.ndarray:
    """Stylised acanthus leaf wave in the border annulus.

    Two overlapping sine-wave components (main wave + fringe) modulated by a
    radial sine envelope that is tall in the centre of the border and tapers
    to zero at both edges — imitating the profile of a raised leaf relief
    without needing external assets.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r = np.sqrt(xs * xs + ys * ys)
    theta = np.arctan2(ys, xs)

    border_outer = outer_r_px - rim_width_px
    border_inner = border_outer - border_width_px
    border_span = max(border_outer - border_inner, 1.0)

    r_norm = (r - border_inner) / border_span
    in_border = (r >= border_inner) & (r <= border_outer)

    # Radial envelope: raised hill centred in border, zero at both edges.
    radial_env = np.sin(np.clip(r_norm, 0.0, 1.0) * math.pi)

    # Angular waves.
    n = float(leaf_count)
    main_wave  = 0.55 + 0.45 * np.sin(theta * n)
    # Fringe half-frequency adds the pointed-tip texture.
    fringe     = 0.20 * np.abs(np.sin(theta * n * 1.5 + math.pi * 0.25))
    # Overlap half-leaves offset by half a period.
    underlayer = 0.35 * (0.5 + 0.5 * np.sin(theta * n + math.pi))

    combined = np.clip(main_wave + fringe + underlayer, 0.0, 1.0)
    out = np.clip(combined * radial_env, 0.0, 1.0)
    out[~in_border] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Border: Greek key (meander / fret)
# ---------------------------------------------------------------------------

# One repeat of the classic fret path in unit-cell coordinates.
# x runs along the band (0 → 1 = one repeat), y across it (0 = inner edge,
# 1 = outer edge).  A continuous baseline links cells; the spiral hook
# rises from it and coils inward (the dead-end is authentic to the motif).
_GREEK_KEY_PATH: tuple = (
    ((0.00, 0.15), (1.00, 0.15)),   # baseline, continuous across repeats
    ((0.10, 0.15), (0.10, 0.85)),   # rise
    ((0.10, 0.85), (0.90, 0.85)),   # top run
    ((0.90, 0.85), (0.90, 0.45)),   # descend
    ((0.90, 0.45), (0.40, 0.45)),   # inward run
    ((0.40, 0.45), (0.40, 0.65)),   # coil up
    ((0.40, 0.65), (0.65, 0.65)),   # coil end
)


def greek_key_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    border_width_px: float,
    key_count: int = 24,
    line_width_frac: float = 0.14,
) -> np.ndarray:
    """Greek key (meander) fret repeated around the border annulus.

    The annulus is unwrapped to per-repeat cell coordinates and the fret
    polyline is rendered as a raised flat-top ridge via point-to-segment
    distance, with a soft edge so the relief has no pixel staircase.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r = np.sqrt(xs * xs + ys * ys)
    theta = np.arctan2(ys, xs)

    border_outer = outer_r_px - rim_width_px
    border_inner = border_outer - border_width_px
    border_span = max(border_outer - border_inner, 1.0)

    in_border = (r >= border_inner) & (r <= border_outer)

    n = max(int(key_count), 4)
    # Cell coordinates in PIXEL units so distance is isotropic: x along the
    # band (one repeat = the arc length of one cell at mid-band radius),
    # y across the band (0 = inner edge).
    mid_r = border_inner + border_span * 0.5
    cell_w_px = 2.0 * math.pi * mid_r / n
    u = ((theta / (2.0 * math.pi) + 0.5) * n) % 1.0     # 0..1 within repeat
    px_x = u * cell_w_px
    px_y = np.clip((r - border_inner) / border_span, 0.0, 1.0) * border_span

    half_lw = max(line_width_frac * border_span * 0.5, 1.0)
    edge = max(half_lw * 0.6, 0.75)  # soft-edge width in px

    min_d = np.full((h, w), np.inf, dtype=np.float32)
    for (x0, y0), (x1, y1) in _GREEK_KEY_PATH:
        ax, ay = x0 * cell_w_px, y0 * border_span
        bx, by = x1 * cell_w_px, y1 * border_span
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-9:
            continue
        t = np.clip(((px_x - ax) * dx + (px_y - ay) * dy) / seg_len2, 0.0, 1.0)
        ddx = px_x - (ax + t * dx)
        ddy = px_y - (ay + t * dy)
        np.minimum(min_d, np.sqrt(ddx * ddx + ddy * ddy), out=min_d)
        # The baseline must also be measured across the cell seam so the
        # ridge doesn't pinch where u wraps 1 → 0.
        if abs(y0 - y1) < 1e-6:  # horizontal segment: test shifted copies
            for shift in (-cell_w_px, cell_w_px):
                t = np.clip(((px_x - (ax + shift)) * dx + (px_y - ay) * dy) / seg_len2, 0.0, 1.0)
                ddx = px_x - (ax + shift + t * dx)
                ddy = px_y - (ay + t * dy)
                np.minimum(min_d, np.sqrt(ddx * ddx + ddy * ddy), out=min_d)

    # Flat-top ridge with cosine-soft shoulders.
    out = np.clip((half_lw + edge - min_d) / edge, 0.0, 1.0)
    out = out * out * (3.0 - 2.0 * out)  # smoothstep shoulder
    out[~in_border] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Border: laurel wreath
# ---------------------------------------------------------------------------

def laurel_wreath_overlay(
    h: int,
    w: int,
    *,
    cx: float,
    cy: float,
    outer_r_px: float,
    rim_width_px: float,
    border_width_px: float,
    leaf_count: int = 36,
) -> np.ndarray:
    """Laurel wreath in the border annulus.

    A central stem ridge with pairs of forward-swept leaves alternating
    above and below it.  Each leaf is a tilted paraboloid ellipse so the
    relief reads as a domed leaf, not a flat cut-out.
    """
    xs = np.arange(w, dtype=np.float32) - cx
    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]
    r = np.sqrt(xs * xs + ys * ys)
    theta = np.arctan2(ys, xs)

    border_outer = outer_r_px - rim_width_px
    border_inner = border_outer - border_width_px
    border_span = max(border_outer - border_inner, 1.0)
    in_border = (r >= border_inner) & (r <= border_outer)

    n = max(int(leaf_count), 6)
    mid_r = border_inner + border_span * 0.5
    cell_w_px = 2.0 * math.pi * mid_r / n

    # Unwrapped pixel coords: x along band within one repeat, y across band
    # centred on the stem (y = 0 at mid-band).
    u = ((theta / (2.0 * math.pi) + 0.5) * n) % 1.0
    px_x = u * cell_w_px
    px_y = (np.clip((r - border_inner) / border_span, 0.0, 1.0) - 0.5) * border_span

    out = np.zeros((h, w), dtype=np.float32)

    # Stem: thin raised ridge along the band centre.
    stem_sigma = border_span * 0.045
    stem = 0.85 * np.exp(-(px_y ** 2) / (2.0 * stem_sigma ** 2))
    np.maximum(out, stem.astype(np.float32), out=out)

    # Leaves: one above + one below the stem per repeat, swept forward.
    leaf_len = cell_w_px * 0.62
    leaf_wid = border_span * 0.26
    tilt = math.radians(38.0)  # forward sweep
    cs, sn = math.cos(tilt), math.sin(tilt)
    for side, x_frac in ((1.0, 0.30), (-1.0, 0.80)):
        # Leaf base sits on the stem; the blade sweeps up-forward.
        base_x = x_frac * cell_w_px
        base_y = 0.0
        # Local coords rotated by the sweep angle (mirrored below stem).
        lx = (px_x - base_x) * cs + (px_y - base_y) * (sn * side)
        ly = -(px_x - base_x) * (sn * side) + (px_y - base_y) * cs
        # Shift so the ellipse centre is mid-blade, base at the stem.
        lx = lx - leaf_len * 0.5
        q = (lx / (leaf_len * 0.5)) ** 2 + (ly / (leaf_wid * 0.5)) ** 2
        leaf = np.clip(1.0 - q, 0.0, 0.999)
        # Paraboloid dome + midrib groove for leafy relief.
        midrib = 1.0 - 0.25 * np.exp(-(ly ** 2) / (2.0 * (leaf_wid * 0.08) ** 2))
        np.maximum(out, (leaf * midrib).astype(np.float32), out=out)

    out = np.clip(out, 0.0, 1.0)
    out[~in_border] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Export-time pattern layers
# ---------------------------------------------------------------------------

def zone_pattern_layer(
    zone: str,
    settings: Dict[str, Any],
    h: int,
    w: int,
    *,
    print_w_mm: float,
    print_h_mm: float,
) -> Optional[np.ndarray]:
    """Standalone decorative relief for one zone pass at export time.

    Unlike :func:`apply_zone_overlays` (which bakes patterns into the master
    heightmap for previewing), this builds the pattern as its OWN engraving
    layer so the exported .lbrn2 zone pass carries the actual relief — even
    when the stored heightmap was rendered before the patterns were chosen.

    ``zone`` is ``"field" | "border" | "rim"``.  Returns a float32 (h, w)
    layer in [0, 1] (1 = surface / no engrave) or ``None`` when no pattern
    is configured for that zone — callers then fall back to masking the
    sculpt heightmap.  The caller composites the layer against white using
    the zone mask, so out-of-zone values here don't need to be exact.

    The layer uses the FULL grayscale range (deepest pattern point = 0.0):
    physical relief depth is set by the zone pass's numPasses, which the
    planner scales by the pattern's depth slider. Pre-scaling the pixels
    too would apply the depth twice.
    """
    if print_w_mm <= 0 or print_h_mm <= 0:
        return None

    cx, cy = w / 2.0, h / 2.0
    px_per_mm = min(w / print_w_mm, h / print_h_mm)
    border_width_mm = float(settings.get("zone_border_width_mm", 1.5))
    rim_width_mm = float(settings.get("zone_rim_width_mm", 0.5))
    outer_r_px = min(print_w_mm, print_h_mm) / 2.0 * px_per_mm
    rim_width_px = rim_width_mm * px_per_mm
    border_width_px = border_width_mm * px_per_mm

    if zone == "field":
        name = str(settings.get("field_pattern", "none")).lower()
        if name == "none":
            return None
        from .backgrounds import generate_pattern
        try:
            overlay = generate_pattern(
                name, w, h,
                scale=float(settings.get("field_pattern_scale", 1.0)),
                angle=float(settings.get("field_pattern_angle", 0.0)),
                seed=int(settings.get("background_seed", 0)),
            )
        except KeyError:
            return None
        return np.clip(overlay, 0.0, 1.0).astype(np.float32)

    if zone == "rim":
        name = str(settings.get("rim_pattern", "none")).lower()
        common = dict(cx=cx, cy=cy, outer_r_px=outer_r_px, rim_width_px=rim_width_px)
        if name == "beaded":
            overlay = beaded_rim_overlay(
                h, w, **common,
                bead_count=int(settings.get("rim_bead_count", 72)), bead_depth=1.0,
            )
        elif name == "reeded":
            overlay = reeded_rim_overlay(
                h, w, **common,
                reed_count=int(settings.get("rim_reed_count", 120)), reed_depth=1.0,
            )
        elif name in ("denticled", "rope", "serrated"):
            fn = {"denticled": denticled_rim_overlay,
                  "rope": rope_rim_overlay,
                  "serrated": serrated_rim_overlay}[name]
            overlay = fn(
                h, w, **common,
                element_count=int(settings.get("rim_element_count", 96)), depth=1.0,
            )
        else:
            return None
        return np.clip(overlay, 0.0, 1.0).astype(np.float32)

    if zone == "border":
        name = str(settings.get("border_pattern", "none")).lower()
        common = dict(
            cx=cx, cy=cy, outer_r_px=outer_r_px,
            rim_width_px=rim_width_px, border_width_px=border_width_px,
        )
        if name == "rope_twist":
            overlay = rope_twist_overlay(
                h, w, **common,
                strand_count=int(settings.get("border_strand_count", 2)),
                twist_periods=int(settings.get("border_twist_periods", 40)),
            )
        elif name == "acanthus_wave":
            overlay = acanthus_wave_overlay(
                h, w, **common,
                leaf_count=int(settings.get("border_leaf_count", 12)),
            )
        elif name == "greek_key":
            overlay = greek_key_overlay(
                h, w, **common,
                key_count=int(settings.get("border_key_count", 24)),
            )
        elif name == "laurel":
            overlay = laurel_wreath_overlay(
                h, w, **common,
                leaf_count=int(settings.get("border_leaf_count", 36)),
            )
        else:
            return None
        return np.clip(overlay, 0.0, 1.0).astype(np.float32)

    return None


# ---------------------------------------------------------------------------
# Composite engine
# ---------------------------------------------------------------------------

def _composite(
    heightmap: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Write ``overlay`` into ``heightmap`` within ``mask``.

    The overlay is interpreted as a raised-surface map (1 = full surface,
    0 = maximal depth).  ``depth`` scales how far below surface the deepest
    overlay point sits — i.e. the floor depth = (1 - depth) in LightBurn
    white-is-surface convention.

    Formula: value = (1 - depth) + overlay * depth
        overlay=1 → value=1.0 (surface, no engraving)
        overlay=0 → value=(1-depth) (maximum depth for this zone)

    The mask is a soft float32 [0,1] blend weight so zone-boundary feathering
    is preserved.
    """
    depth = float(np.clip(depth, 0.0, 1.0))
    target = (1.0 - depth) + overlay * depth
    blended = heightmap * (1.0 - mask) + target * mask
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def apply_zone_overlays(
    heightmap: np.ndarray,
    settings: Dict[str, Any],
    subject_alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Apply rim, border and field overlays to ``heightmap`` in-place (returns new array).

    ``subject_alpha`` (0=background, 1=subject; any resolution) protects the
    subject relief: the field pattern is excluded wherever the subject sits,
    mirroring the device/field mask split the exporter applies per pass.

    Reads the following keys from ``settings``:

    Geometry (required — skipped if zone_width_mm == 0):
        zone_width_mm, zone_height_mm, zone_shape,
        zone_border_width_mm, zone_rim_width_mm

    Field:
        field_pattern          — "none" | "radial_lines" | "guilloche" | ...
        field_pattern_scale    — generator scale (default 1.0)
        field_pattern_angle    — rotation degrees (default 0.0)
        field_pattern_depth    — 0-1, how deep the field floor engraves (default 0.70)

    Rim:
        rim_pattern            — "none" | "beaded" | "reeded" | "denticled" |
                                 "rope" | "serrated"
        rim_bead_count         — integer, beads around circumference (default 72)
        rim_reed_count         — integer, reeds around circumference (default 120)
        rim_element_count      — integer, denticle/rope/serration repeats (default 96)
        rim_pattern_depth      — peak element height in [0,1] (default 1.0)

    Border:
        border_pattern         — "none" | "rope_twist" | "acanthus_wave" |
                                 "greek_key" | "laurel"
        border_pattern_depth   — 0-1 depth (default 0.85)
        border_strand_count    — rope strands (default 2)
        border_twist_periods   — rope full-cycle count (default 40)
        border_leaf_count      — acanthus/laurel leaf repeats (default 12 / 36)
        border_key_count       — greek key repeats (default 24)
    """
    zone_w = float(settings.get("zone_width_mm", 0.0))
    zone_h_mm = float(settings.get("zone_height_mm", 0.0))
    if zone_w <= 0.0 or zone_h_mm <= 0.0:
        return heightmap  # geometry not configured — no-op

    h, w = heightmap.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # px_per_mm from the smaller axis so the target fits the canvas.
    px_per_mm = min(w / zone_w, h / zone_h_mm)

    shape = str(settings.get("zone_shape", "circle")).lower()
    border_width_mm = float(settings.get("zone_border_width_mm", 1.5))
    rim_width_mm    = float(settings.get("zone_rim_width_mm",    0.5))

    outer_r_px     = min(zone_w, zone_h_mm) / 2.0 * px_per_mm
    rim_width_px   = rim_width_mm   * px_per_mm
    border_width_px = border_width_mm * px_per_mm

    # ------------------------------------------------------------------
    # Zone masks (lazy import to avoid circular dependency)
    # ------------------------------------------------------------------
    from .zones import zone_masks_from_geometry

    zones = zone_masks_from_geometry(
        h, w, shape,
        print_w_mm=zone_w,
        print_h_mm=zone_h_mm,
        px_per_mm=px_per_mm,
        border_width_mm=border_width_mm,
        rim_width_mm=rim_width_mm,
    )

    result = heightmap.copy()

    # ------------------------------------------------------------------
    # 1. Field pattern
    # ------------------------------------------------------------------
    field_pattern = str(settings.get("field_pattern", "none")).lower()
    if field_pattern != "none":
        from .backgrounds import generate_pattern
        try:
            overlay = generate_pattern(
                field_pattern, w, h,
                scale=float(settings.get("field_pattern_scale", 1.0)),
                angle=float(settings.get("field_pattern_angle", 0.0)),
                seed=int(settings.get("background_seed", 0)),
            )
        except KeyError:
            overlay = None
        if overlay is not None:
            depth = float(settings.get("field_pattern_depth", 0.70))
            field_mask = zones["field"]
            if subject_alpha is not None:
                alpha = np.clip(subject_alpha.astype(np.float32, copy=False), 0.0, 1.0)
                if alpha.shape != (h, w):
                    from PIL import Image as _PILImage
                    alpha = np.asarray(
                        _PILImage.fromarray((alpha * 255.0 + 0.5).astype(np.uint8), mode="L")
                        .resize((w, h), _PILImage.LANCZOS),
                        dtype=np.float32,
                    ) / 255.0
                field_mask = np.clip(field_mask * (1.0 - alpha), 0.0, 1.0)
            result = _composite(result, overlay, field_mask, depth)

    # ------------------------------------------------------------------
    # 2. Rim pattern
    # ------------------------------------------------------------------
    rim_pattern = str(settings.get("rim_pattern", "none")).lower()
    if rim_pattern == "beaded":
        overlay = beaded_rim_overlay(
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            bead_count=int(settings.get("rim_bead_count", 72)),
            bead_depth=float(settings.get("rim_pattern_depth", 1.0)),
        )
        # Rim beads fully replace the rim zone.
        result = _composite(result, overlay, zones["rim"], depth=1.0)

    elif rim_pattern == "reeded":
        overlay = reeded_rim_overlay(
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            reed_count=int(settings.get("rim_reed_count", 120)),
            reed_depth=float(settings.get("rim_pattern_depth", 1.0)),
        )
        result = _composite(result, overlay, zones["rim"], depth=1.0)

    elif rim_pattern in ("denticled", "rope", "serrated"):
        _RIM_FNS = {
            "denticled": denticled_rim_overlay,
            "rope": rope_rim_overlay,
            "serrated": serrated_rim_overlay,
        }
        overlay = _RIM_FNS[rim_pattern](
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            element_count=int(settings.get("rim_element_count", 96)),
            depth=float(settings.get("rim_pattern_depth", 1.0)),
        )
        result = _composite(result, overlay, zones["rim"], depth=1.0)

    # ------------------------------------------------------------------
    # 3. Border pattern
    # ------------------------------------------------------------------
    border_pattern = str(settings.get("border_pattern", "none")).lower()
    border_depth   = float(settings.get("border_pattern_depth", 0.85))

    if border_pattern == "rope_twist":
        overlay = rope_twist_overlay(
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            border_width_px=border_width_px,
            strand_count=int(settings.get("border_strand_count", 2)),
            twist_periods=int(settings.get("border_twist_periods", 40)),
        )
        result = _composite(result, overlay, zones["border"], border_depth)

    elif border_pattern == "acanthus_wave":
        overlay = acanthus_wave_overlay(
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            border_width_px=border_width_px,
            leaf_count=int(settings.get("border_leaf_count", 12)),
        )
        result = _composite(result, overlay, zones["border"], border_depth)

    elif border_pattern == "greek_key":
        overlay = greek_key_overlay(
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            border_width_px=border_width_px,
            key_count=int(settings.get("border_key_count", 24)),
        )
        result = _composite(result, overlay, zones["border"], border_depth)

    elif border_pattern == "laurel":
        overlay = laurel_wreath_overlay(
            h, w,
            cx=cx, cy=cy,
            outer_r_px=outer_r_px,
            rim_width_px=rim_width_px,
            border_width_px=border_width_px,
            leaf_count=int(settings.get("border_leaf_count", 36)),
        )
        result = _composite(result, overlay, zones["border"], border_depth)

    return result
