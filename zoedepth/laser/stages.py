"""Pass planner: turn a finished heightmap stack into ordered engraving passes.

The user opts into any subset of the canonical pass kinds; the planner
returns an ordered :class:`EngravingPass` list with verbatim ``ColorEntry``
machine parameters lifted from the chosen :class:`MaterialProfile`.

Pass kinds (see ``IMPLEMENTATION_PLAN.md`` §4):

1. ``pre_clean`` — defocused light pass to remove oxidation / oils.
2. ``form``     — the bulk relief itself (the heightmap PNG).
3. ``cleanup``  — narrow contour around the form, suppresses chatter.
4. ``detail``   — high-frequency micro-relief (Stage-B detail PNG).
5. ``shading``  — soft photometric shading where appropriate.
6. ``polish``   — final dithered surface pass.
7. ``color_*``  — one pass per discovered color region (one color card row each).
8. ``signature``— optional vector text / monogram, line-engraving style.

Every pass is an opt-in: the planner accepts a ``user_toggles`` map
(`pass_kind -> bool`) and silently drops the disabled ones.

Color passes are auto-derived from a ``mask_per_color`` mapping (built by
the upcoming color-picker stage); each entry is paired with the
``ColorEntry`` of the same name in the active profile so the laser
parameters travel verbatim into the exported ``.lbrn2``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    "PASS_KIND_CLEANUP",
    "PASS_KIND_DETAIL",
    "PASS_KIND_SHADING",
    "PASS_KIND_POLISH",
    "PASS_KIND_COLOR_PREFIX",
    "PASS_KIND_SIGNATURE",
]


# ----------------------------------------------------------- pass-kind constants

PASS_KIND_PRE_CLEAN = "pre_clean"
PASS_KIND_FORM = "form"
PASS_KIND_CLEANUP = "cleanup"
PASS_KIND_DETAIL = "detail"
PASS_KIND_SHADING = "shading"
PASS_KIND_POLISH = "polish"
PASS_KIND_SIGNATURE = "signature"
# Per-color pass keys are formed as ``{PASS_KIND_COLOR_PREFIX}{color_name}``
# (e.g. ``"color:C03"``). Keeps every key unique while letting the planner
# group them via ``startswith``.
PASS_KIND_COLOR_PREFIX = "color:"

# Canonical execution order. Color passes are inserted between SHADING and
# POLISH at plan time (one entry per active color, in their card index).
DEFAULT_PASS_ORDER: tuple[str, ...] = (
    PASS_KIND_PRE_CLEAN,
    PASS_KIND_FORM,
    PASS_KIND_CLEANUP,
    PASS_KIND_DETAIL,
    PASS_KIND_SHADING,
    # color passes go here
    PASS_KIND_POLISH,
    PASS_KIND_SIGNATURE,
)


# --------------------------------------------------------------- data classes

@dataclass(frozen=True)
class EngravingPass:
    """One pass in the planned stack.

    Carries everything the LBRN writer needs:

    * a stable ``id`` (used as the LightBurn layer index when writing),
    * a ``mask`` raster (float32 ``[0, 1]``) describing where the pass fires,
    * the verbatim ``cut_setting`` lifted from the active material profile,
    * ``depends_on`` — pass ids that must precede this one (used by the UI to
      enforce sensible toggle interactions, e.g. Cleanup follows Form).
    """

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
    PASS_KIND_PRE_CLEAN: ("C00", "Defocused oxidation/oil burn-off.", ()),
    PASS_KIND_FORM:      ("C01", "Bulk relief (heightmap).", ()),
    PASS_KIND_CLEANUP:   ("C02", "Edge contour to suppress chatter.", (PASS_KIND_FORM,)),
    PASS_KIND_DETAIL:    ("C03", "High-frequency micro-relief.", (PASS_KIND_FORM,)),
    PASS_KIND_SHADING:   ("C04", "Soft photometric shading.", (PASS_KIND_FORM,)),
    PASS_KIND_POLISH:    ("C05", "Final dithered surface pass.", (PASS_KIND_FORM,)),
    PASS_KIND_SIGNATURE: ("C06", "Vector signature / monogram.", ()),
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
        Float32 ``H×W`` array used to size any default masks and to validate
        user-supplied masks.
    profile
        The :class:`MaterialProfile` whose ``ColorEntry`` rows feed cut
        parameters into each pass.
    user_toggles
        Map ``pass_kind -> bool``. Missing kinds default to ``True``. To
        toggle a per-color pass off, use ``f"{PASS_KIND_COLOR_PREFIX}{name}"``.
    masks
        Optional per-kind masks (float32 in ``[0, 1]``). Keys are the
        ``PASS_KIND_*`` constants. Missing kinds get an all-ones mask.
    mask_per_color
        Map ``color_name -> mask`` produced by the color-picker stage. One
        :class:`EngravingPass` of kind ``color:<name>`` is emitted per entry,
        in the source profile's index order. Color names without a matching
        :class:`ColorEntry` in ``profile`` are silently dropped.
    kind_color_overrides
        Override for the per-kind default color names (e.g. point Form at
        ``"C09"`` instead of the default ``"C01"``).
    """
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must be 2-D; got shape {heightmap.shape}")

    toggles = dict(user_toggles or {})
    overrides = dict(kind_color_overrides or {})

    ordered: List[EngravingPass] = []
    for kind in DEFAULT_PASS_ORDER:
        if kind == PASS_KIND_POLISH:
            # Insert color passes immediately before the polish pass.
            ordered.extend(_plan_color_passes(
                heightmap=heightmap,
                profile=profile,
                mask_per_color=mask_per_color or {},
                toggles=toggles,
            ))
        if not toggles.get(kind, True):
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
    # Sort by the entry's LightBurn index so passes execute in card order.
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
