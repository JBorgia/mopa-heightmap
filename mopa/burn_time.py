"""Per-pass burn-time estimator for an engraving stack.

Given a :class:`mopa.stages.PassPlan` plus the physical print
size, estimate how long each pass takes and the total job duration. The
arithmetic is the standard galvo-raster model:

    pixels_per_line   = scan_width_mm / line_interval_mm
    seconds_per_line  = scan_width_mm / scan_speed_mm_s
    lines             = scan_height_mm / line_interval_mm
    seconds           = lines × seconds_per_line × pass_count

We multiply this base by the *active fraction* of each pass — the mean
mask value — so a Cleanup pass that only fires on the silhouette ring
isn't billed at full subject area. (Galvos still traverse the empty
scanlines in many controllers, but most modern MOPAs skip dark areas
above a threshold, so this is the realistic figure.)

The output is intentionally a `BurnEstimate` dataclass — the caller
decides how to format the times (CLI prints minutes:seconds, the API
returns milliseconds, the wizard pane shows "≈ 3 h 12 min").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .stages import EngravingPass, PassPlan


__all__ = [
    "BurnEstimate",
    "PassBurnEstimate",
    "estimate_burn_time",
    "format_seconds",
    "DEFAULT_PASS_COUNT",
]


# Some MOPA workflows fire the same color/pass parameters multiple times
# to deepen the cut. The pass-stack we generate uses one pass per logical
# stage, so the multiplier defaults to 1; callers that bake in multi-pass
# physics (e.g. depth budgets) should override per-pass.
DEFAULT_PASS_COUNT: int = 1


@dataclass(frozen=True)
class PassBurnEstimate:
    pass_id: str
    kind: str
    name: str
    seconds: float
    active_fraction: float          # mean(mask), 0..1
    pass_count: int


@dataclass(frozen=True)
class BurnEstimate:
    width_mm: float
    height_mm: float
    passes: Tuple[PassBurnEstimate, ...]

    @property
    def total_seconds(self) -> float:
        return float(sum(p.seconds for p in self.passes))


def _pass_seconds(
    ep: EngravingPass,
    *,
    width_mm: float,
    height_mm: float,
    pass_count: int,
) -> float:
    cs = ep.cut_setting
    speed_mm_s = float(cs.speed) if cs.speed > 0 else 1.0
    line_interval_mm = float(cs.interval) if cs.interval > 0 else 0.025
    seconds_per_line = width_mm / speed_mm_s
    lines = max(1.0, height_mm / line_interval_mm)
    base_seconds = lines * seconds_per_line * max(1, int(pass_count))
    active = float(np.clip(np.mean(ep.mask), 0.0, 1.0))
    return base_seconds * active


def estimate_burn_time(
    plan: PassPlan,
    *,
    width_mm: float,
    height_mm: float,
    pass_count_overrides: Optional[dict[str, int]] = None,
) -> BurnEstimate:
    """Compute per-pass and total burn time for ``plan`` at ``width_mm × height_mm``.

    ``pass_count_overrides`` lets callers specify multi-pass deepening for
    individual pass kinds (e.g. ``{"form": 32, "polish": 1}``). Anything
    not in the override map gets ``DEFAULT_PASS_COUNT``.
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(
            f"width_mm and height_mm must be positive; got {width_mm} × {height_mm}"
        )

    overrides = pass_count_overrides or {}
    rows: List[PassBurnEstimate] = []
    for ep in plan.passes:
        n = int(overrides.get(ep.kind, DEFAULT_PASS_COUNT))
        seconds = _pass_seconds(ep, width_mm=width_mm, height_mm=height_mm, pass_count=n)
        rows.append(PassBurnEstimate(
            pass_id=ep.id,
            kind=ep.kind,
            name=ep.name,
            seconds=seconds,
            active_fraction=float(np.clip(np.mean(ep.mask), 0.0, 1.0)),
            pass_count=n,
        ))
    return BurnEstimate(
        width_mm=float(width_mm),
        height_mm=float(height_mm),
        passes=tuple(rows),
    )


def format_seconds(s: float) -> str:
    """Format ``s`` seconds as ``H h M m S s`` / ``M m S s`` / ``S.S s``."""
    if s < 0:
        return "0 s"
    if s < 60:
        return f"{s:.1f} s"
    minutes, seconds = divmod(int(round(s)), 60)
    if minutes < 60:
        return f"{minutes} m {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} m {seconds:02d} s"
