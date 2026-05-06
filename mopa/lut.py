"""Calibration LUT — gray→depth mapping fit from operator measurements.

Workflow:
    1. Operator burns the calibration ramp (`preview.create_calibration_ramp()`)
       which prints 11 patches at gray levels 0,25,51,...,255.
    2. Operator measures actual depth (microns) at each patch with a depth gauge
       or microscope and types those numbers in.
    3. We fit a monotonic function `gray -> measured_depth_um`.
    4. The LUT is used in two ways:
         a) Diagnostic: show "your machine maps 50% gray to 38 µm".
         b) Compensation: when the operator sets a target depth budget, we
            invert the LUT so the engraver actually delivers a perceptually
            linear depth ramp.

Profile YAML carries this block (all numbers; YAML-friendly):

    calibration_lut:
      note: "150mm lens, brass, measured 2026-04-21"
      max_depth_um: 120.0          # optional; the deepest measured patch
      samples:
        - [0,   0.0]
        - [25,  4.5]
        - [51,  10.2]
        ...
        - [255, 120.0]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence, Tuple

import numpy as np


_DEFAULT_RAMP_GRAY: Tuple[int, ...] = (0, 25, 51, 76, 102, 128, 153, 178, 204, 229, 255)


@dataclass
class CalibrationLUT:
    samples: List[Tuple[float, float]]  # (gray_0_255, depth_um)
    note: str = ""
    max_depth_um: float | None = None

    # ------------------------------------------------------------- factories
    @classmethod
    def from_measurements(
        cls,
        depths_um: Sequence[float],
        gray_levels: Sequence[float] | None = None,
        *,
        note: str = "",
    ) -> "CalibrationLUT":
        """Build a LUT from an ordered list of measured depths.

        If `gray_levels` is omitted, assumes the standard 11-patch ramp.
        """
        if gray_levels is None:
            gray_levels = _DEFAULT_RAMP_GRAY[: len(depths_um)]
        if len(gray_levels) != len(depths_um):
            raise ValueError(
                f"gray_levels ({len(gray_levels)}) and depths_um ({len(depths_um)}) "
                "must have the same length"
            )
        if len(depths_um) < 2:
            raise ValueError("Need at least 2 sample points to fit a LUT.")
        samples = list(zip([float(g) for g in gray_levels], [float(d) for d in depths_um]))
        samples.sort(key=lambda p: p[0])
        return cls(samples=samples, note=note, max_depth_um=max(d for _, d in samples))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibrationLUT":
        raw = payload.get("samples") or []
        samples: List[Tuple[float, float]] = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                samples.append((float(entry[0]), float(entry[1])))
            elif isinstance(entry, Mapping):
                samples.append((float(entry["gray"]), float(entry["depth_um"])))
            else:
                raise ValueError(f"Bad LUT sample entry: {entry!r}")
        if len(samples) < 2:
            raise ValueError("calibration_lut.samples needs at least 2 entries.")
        samples.sort(key=lambda p: p[0])
        return cls(
            samples=samples,
            note=str(payload.get("note", "")),
            max_depth_um=(
                float(payload["max_depth_um"]) if payload.get("max_depth_um") is not None
                else max(d for _, d in samples)
            ),
        )

    def to_dict(self) -> dict:
        return {
            "note": self.note,
            "max_depth_um": self.max_depth_um,
            "samples": [[float(g), float(d)] for g, d in self.samples],
        }

    # --------------------------------------------------------------- arrays
    def _arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        g = np.asarray([p[0] for p in self.samples], dtype=np.float64)
        d = np.asarray([p[1] for p in self.samples], dtype=np.float64)
        # Force monotonic non-decreasing in depth — physical reality.
        d = np.maximum.accumulate(d)
        return g, d

    # --------------------------------------------------------------- queries
    def gray_to_depth_um(self, gray_0_255: float | np.ndarray) -> np.ndarray:
        g, d = self._arrays()
        return np.interp(np.asarray(gray_0_255, dtype=np.float64), g, d)

    def depth_um_to_gray(self, depth_um: float | np.ndarray) -> np.ndarray:
        g, d = self._arrays()
        return np.interp(np.asarray(depth_um, dtype=np.float64), d, g)

    # --------------------------------------------------------------- apply
    def apply(
        self,
        heightmap: np.ndarray,
        *,
        target_depth_um: float | None = None,
    ) -> np.ndarray:
        """Re-map a [0,1] heightmap so requested depth is what the laser delivers.

        For each input pixel value `v` (in [0,1] where 1=surface, 0=deepest):
            requested_depth_um = (1 - v) * target_depth_um
            new_gray_0_255 = depth_um_to_gray(requested_depth_um)
            new_v = 1 - new_gray_0_255 / 255

        If `target_depth_um` is None we use `max_depth_um` from the LUT itself.
        """
        if heightmap.ndim != 2:
            raise ValueError("heightmap must be 2D")
        if target_depth_um is None:
            target_depth_um = self.max_depth_um
        if target_depth_um is None or target_depth_um <= 0:
            return heightmap.astype(np.float32)

        v = np.clip(heightmap.astype(np.float64), 0.0, 1.0)
        requested_depth = (1.0 - v) * float(target_depth_um)
        new_gray = self.depth_um_to_gray(requested_depth)
        new_v = 1.0 - (new_gray / 255.0)
        return np.clip(new_v, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------- helpers
def lut_from_profile(profile_data: Mapping[str, object]) -> CalibrationLUT | None:
    """Return a CalibrationLUT if the profile carries one, else None."""
    raw = profile_data.get("calibration_lut")
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("profile.calibration_lut must be a mapping")
    return CalibrationLUT.from_dict(raw)
