"""Engraving-time estimator.

Approximates total burn time for an `n_pixels_high × n_pixels_wide` heightmap
processed in 3D Sliced mode on a galvo head. The model is intentionally
simple: it captures the dominant cost (line scans), not jump moves or
acceleration, and is good enough to keep the operator from queuing a 12-hour
job by accident.

Inputs come from the active material profile's `lightburn_starting_point` block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class EngraveEstimate:
    seconds: float
    minutes: float
    hours: float
    line_count: int
    notes: list[str]

    def human(self) -> str:
        if self.hours >= 1.0:
            return f"~{self.hours:.1f} h ({self.minutes:.0f} min)"
        if self.minutes >= 1.0:
            return f"~{self.minutes:.1f} min"
        return f"~{self.seconds:.0f} s"


def _get_first(mapping: Mapping[str, object], *keys: str, default=None):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return default


def estimate_engrave_time(
    image_height_px: int,
    image_width_px: int,
    *,
    physical_height_mm: float,
    physical_width_mm: float,
    lightburn_starting_point: Mapping[str, object],
    overhead_factor: float = 1.15,
) -> EngraveEstimate:
    """Estimate burn time given image dims, physical size, and a profile's cut block.

    Accepts both the bare names (`speed`, `line_interval`) and the unit-suffixed
    names used by the shipped MOPA YAML profiles (`speed_mm_s`, `line_interval_mm`).
    """
    notes: list[str] = []

    speed_mm_s = float(_get_first(
        lightburn_starting_point, "speed_mm_s", "speed", default=2000.0,
    ))
    passes = int(_get_first(lightburn_starting_point, "passes", default=1))
    line_interval_mm = float(_get_first(
        lightburn_starting_point, "line_interval_mm", "line_interval", default=0.04,
    ))
    angle_increment = float(_get_first(
        lightburn_starting_point, "angle_increment", default=0.0,
    ))

    if speed_mm_s <= 0:
        raise ValueError("speed must be > 0 mm/s")
    if line_interval_mm <= 0:
        raise ValueError("line_interval must be > 0 mm")

    line_count_per_pass = max(int(round(physical_height_mm / line_interval_mm)), 1)
    total_lines = line_count_per_pass * max(passes, 1)
    line_length_mm = max(physical_width_mm, 1e-3)

    seconds = (total_lines * line_length_mm) / speed_mm_s
    seconds *= max(overhead_factor, 1.0)

    if angle_increment > 0:
        notes.append(
            f"angle_increment={angle_increment}° — actual time may exceed estimate by ~5–15%."
        )
    if passes >= 100:
        notes.append(f"{passes} passes — confirm depth budget before running.")
    if seconds > 3600:
        notes.append("Job exceeds 1 hour. Verify cooling and material clamping.")

    return EngraveEstimate(
        seconds=seconds,
        minutes=seconds / 60.0,
        hours=seconds / 3600.0,
        line_count=total_lines,
        notes=notes,
    )


def estimate_from_profile(
    image_size: tuple[int, int],
    physical_size_mm: tuple[float, float],
    profile_data: Mapping[str, object],
) -> EngraveEstimate:
    """Convenience wrapper that pulls the cut block out of a loaded profile."""
    cut = profile_data.get("lightburn_starting_point") or {}
    if not isinstance(cut, Mapping):
        raise ValueError("profile.lightburn_starting_point must be a mapping")
    h_px, w_px = image_size
    h_mm, w_mm = physical_size_mm
    return estimate_engrave_time(
        h_px, w_px,
        physical_height_mm=h_mm,
        physical_width_mm=w_mm,
        lightburn_starting_point=cut,
    )
