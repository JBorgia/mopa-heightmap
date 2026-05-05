"""Pass planner: turn a sculptok heightmap + optional refinement masks into ordered engraving passes.

The product model: sculptok produces ONE depth heightmap that becomes the
3D-Sliced bitmap layer (kind ``form``). Refinement passes add separate
physical features on top of the carved relief — they do NOT subdivide
the heightmap depth budget.

Pass kinds:

1. ``pre_clean``  — defocused light pass to remove oxidation / oils. Opt-in.
2. ``form``       — the sculptok depth bitmap (.lbrn2 3D Sliced layer).
3. ``color:*``    — one pass per color cluster (LAB k-means on the photo).
4. ``photo_tonal``— low-power dithered photo-luminance overlay. Opt-in.
5. ``signature``  — small text rendered into a corner. Opt-in via text.

Every pass is opt-in via ``user_toggles``; ``form`` is the only one that
defaults to enabled because it carries the depth budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .lightburn_cards import ColorEntry, MaterialProfile


__all__ = [
    "EngravingPass",
    "PassPlan",
    "plan_passes",
    "DEFAULT_PASS_ORDER",
    "PASS_KIND_FORM",
    "PASS_KIND_PRE_CLEAN",
    "PASS_KIND_PHOTO_TONAL",
    "PASS_KIND_COLOR_PREFIX",
    "PASS_KIND_SIGNATURE",
]


# ----------------------------------------------------------- pass-kind constants

PASS_KIND_PRE_CLEAN = "pre_clean"
PASS_KIND_FORM = "form"
PASS_KIND_PHOTO_TONAL = "photo_tonal"
PASS_KIND_SIGNATURE = "signature"
# Per-color pass keys are formed as ``{PASS_KIND_COLOR_PREFIX}{color_name}``
# (e.g. ``"color:C03"``).
PASS_KIND_COLOR_PREFIX = "color:"

# Canonical execution order. Color passes are inserted between FORM and
# PHOTO_TONAL at plan time. Signature comes last so the corner mark sits
# on top of everything else.
DEFAULT_PASS_ORDER: tuple[str, ...] = (
    PASS_KIND_PRE_CLEAN,
    PASS_KIND_FORM,
    # color passes go here
    PASS_KIND_PHOTO_TONAL,
    PASS_KIND_SIGNATURE,
)


# --------------------------------------------------------------- data classes

@dataclass(frozen=True)
class EngravingPass:
    """One pass in the planned stack."""

    id: str
    kind: str
    name: str
    mask: np.ndarray
    cut_setting: ColorEntry
    enabled: bool = True
    depends_on: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class PassPlan:
    """Ordered, enabled-only view returned by :func:`plan_passes`."""

    profile: MaterialProfile
    passes: tuple[EngravingPass, ...]

    def by_kind(self, kind: str) -> tuple[EngravingPass, ...]:
        return tuple(p for p in self.passes if p.kind == kind)

    def by_id(self, pass_id: str) -> Optional[EngravingPass]:
        for p in self.passes:
            if p.id == pass_id:
                return p
        return None


# --------------------------------------------------------------- helpers

def _full_mask_like(reference: np.ndarray) -> np.ndarray:
    """Return an all-ones float32 mask sized like ``reference``."""
    return np.ones(reference.shape[:2], dtype=np.float32)


def _resolve_color_entry(
    profile: MaterialProfile,
    name: str,
) -> Optional[ColorEntry]:
    """Look up a color entry by name; return None if absent (skipped pass)."""
    return profile.by_name.get(name)


def _build_kind_pass(
    *,
    kind: str,
    profile: MaterialProfile,
    color_name: str,
    heightmap: np.ndarray,
    masks: Mapping[str, np.ndarray] | None,
    note: str,
    depends_on: Sequence[str] = (),
) -> Optional[EngravingPass]:
    """Build a single non-color pass; returns None if its color is missing."""
    entry = _resolve_color_entry(profile, color_name)
    if entry is None:
        return None
    mask = (masks or {}).get(kind)
    if mask is None:
        mask = _full_mask_like(heightmap)
    if mask.shape[:2] != heightmap.shape[:2]:
        raise ValueError(
            f"mask for {kind!r} has shape {mask.shape}; expected {heightmap.shape}"
        )
    return EngravingPass(
        id=kind,
        kind=kind,
        name=color_name,
        mask=mask.astype(np.float32),
        cut_setting=entry,
        enabled=True,
        depends_on=tuple(depends_on),
        note=note,
    )


# ------------------------------------------------------------------- planner

# Kind -> (which color entry to lift from the profile, default note,
# default upstream-pass dependencies). The color names here are the
# LightBurn card defaults; if the user's profile uses different names
# they should override via ``kind_color_overrides``.
_KIND_DEFAULTS: Dict[str, tuple[str, str, tuple[str, ...]]] = {
    PASS_KIND_PRE_CLEAN:   ("C00", "Defocused oxidation/oil burn-off.", ()),
    PASS_KIND_FORM:        ("C01", "Sculptok depth — 3D-Sliced bitmap.", ()),
    PASS_KIND_PHOTO_TONAL: ("C07", "Photo-derived tonal overlay.", (PASS_KIND_FORM,)),
    PASS_KIND_SIGNATURE:   ("C06", "Vector signature / monogram.", ()),
}


def plan_passes(
    *,
    heightmap: np.ndarray,
    profile: MaterialProfile,
    user_toggles: Mapping[str, bool] | None = None,
    masks: Mapping[str, np.ndarray] | None = None,
    mask_per_color: Mapping[str, np.ndarray] | None = None,
    kind_color_overrides: Mapping[str, str] | None = None,
) -> PassPlan:
    """Build the ordered :class:`EngravingPass` stack.

    Parameters
    ----------
    heightmap
        Float32 ``H×W`` array used to size any default masks.
    profile
        The :class:`MaterialProfile` whose ``ColorEntry`` rows feed cut
        parameters into each pass.
    user_toggles
        Map ``pass_kind -> bool``. Missing kinds default to ``True`` for
        ``form`` and ``False`` for everything else (refinement passes
        are opt-in).
    masks
        Optional per-kind masks (float32 in ``[0, 1]``). Missing kinds
        get an all-ones mask, which is correct for the depth pass
        (``form``) since the sculptok PNG IS the engraving target.
    mask_per_color
        Map ``color_name -> mask`` for the LAB k-means color clusters.
    kind_color_overrides
        Override for the per-kind default color names.
    """
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must be 2-D; got shape {heightmap.shape}")

    toggles = dict(user_toggles or {})
    overrides = dict(kind_color_overrides or {})

    # Refinement passes are opt-in; only ``form`` defaults to enabled.
    def _enabled(kind: str) -> bool:
        if kind in toggles:
            return bool(toggles[kind])
        return kind == PASS_KIND_FORM

    ordered: List[EngravingPass] = []
    for kind in DEFAULT_PASS_ORDER:
        if kind == PASS_KIND_PHOTO_TONAL:
            # Insert color passes immediately before the photo-tonal pass.
            ordered.extend(_plan_color_passes(
                heightmap=heightmap,
                profile=profile,
                mask_per_color=mask_per_color or {},
                toggles=toggles,
            ))
        if not _enabled(kind):
            continue
        color_name, note, depends_on = _KIND_DEFAULTS[kind]
        color_name = overrides.get(kind, color_name)
        built = _build_kind_pass(
            kind=kind, profile=profile, color_name=color_name,
            heightmap=heightmap, masks=masks, note=note, depends_on=depends_on,
        )
        if built is not None:
            ordered.append(built)
    return PassPlan(profile=profile, passes=tuple(ordered))


def _plan_color_passes(
    *,
    heightmap: np.ndarray,
    profile: MaterialProfile,
    mask_per_color: Mapping[str, np.ndarray],
    toggles: Mapping[str, bool],
) -> List[EngravingPass]:
    """Emit one EngravingPass per color, in profile.index order."""
    out: List[EngravingPass] = []
    for entry in profile.entries:
        if entry.name not in mask_per_color:
            continue
        toggle_key = f"{PASS_KIND_COLOR_PREFIX}{entry.name}"
        if not toggles.get(toggle_key, True):
            continue
        mask = mask_per_color[entry.name]
        if mask.shape[:2] != heightmap.shape[:2]:
            raise ValueError(
                f"color mask {entry.name!r} has shape {mask.shape}; "
                f"expected {heightmap.shape}"
            )
        out.append(EngravingPass(
            id=toggle_key,
            kind=toggle_key,
            name=entry.name,
            mask=mask.astype(np.float32),
            cut_setting=entry,
            enabled=True,
            depends_on=(PASS_KIND_FORM,),
            note=f"Color region {entry.name}.",
        ))
    return out
