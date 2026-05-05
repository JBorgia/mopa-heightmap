"""Compose a bulk depth field with a micro-relief integrated from normals.

Phase 3 of ``IMPLEMENTATION_PLAN.md``: blend the form-giving depth map
(ZoeDepth / Depth-Anything-V2) with the high-frequency micro-relief
recovered by integrating surface normals through Frankot–Chellappa.

The combine is a convex combination after both fields are normalised into
``[0, 1]``:

.. math::

    H = w_{\\text{bulk}} \\cdot \\hat{D} + w_{\\text{micro}} \\cdot \\hat{R},
    \\quad w_{\\text{bulk}} + w_{\\text{micro}} = 1

Exposed in the UI as a single "Detail vs. Form" slider where 0 = pure form
(legacy behaviour) and 1 = pure micro-relief (rare; useful for shading the
heightmap directly from the photometric channel).
"""
from __future__ import annotations

import numpy as np


__all__ = [
    "compose_relief",
    "DEFAULT_BULK_WEIGHT",
    "DEFAULT_MICRO_WEIGHT",
    "DETAIL_FORM_SLIDER_MIN",
    "DETAIL_FORM_SLIDER_MAX",
    "EPS_RANGE",
    "normalise_unit",
    "detail_slider_to_weights",
]


# Default split favours form over micro-relief because over-emphasised
# micro-relief tends to "etched-noise" backgrounds. 0.7/0.3 matches the
# value committed in IMPLEMENTATION_PLAN.md §Phase 3.
DEFAULT_BULK_WEIGHT: float = 0.7
DEFAULT_MICRO_WEIGHT: float = 1.0 - DEFAULT_BULK_WEIGHT

# UI slider bounds for the Detail-vs-Form control.
DETAIL_FORM_SLIDER_MIN: float = 0.0
DETAIL_FORM_SLIDER_MAX: float = 1.0

# Floor used when normalising a constant field (max == min) so we don't
# divide by zero. Picks float32-eps-ish to match downstream tolerance.
EPS_RANGE: float = 1e-6


def normalise_unit(field: np.ndarray) -> np.ndarray:
    """Affine-rescale ``field`` to ``[0, 1]``. Constant inputs become zero."""
    arr = np.asarray(field, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    span = hi - lo
    if span <= EPS_RANGE:
        return np.zeros_like(arr)
    return ((arr - lo) / span).astype(np.float32)


def detail_slider_to_weights(slider: float) -> tuple[float, float]:
    """Map a single 0..1 "Detail vs. Form" slider to ``(w_bulk, w_micro)``."""
    s = float(np.clip(slider, DETAIL_FORM_SLIDER_MIN, DETAIL_FORM_SLIDER_MAX))
    return (1.0 - s, s)


def compose_relief(
    bulk_depth: np.ndarray,
    micro_relief: np.ndarray,
    *,
    bulk_weight: float = DEFAULT_BULK_WEIGHT,
    micro_weight: float = DEFAULT_MICRO_WEIGHT,
) -> np.ndarray:
    """Convex-combine bulk depth with micro-relief into one heightmap.

    Both inputs are independently rescaled to ``[0, 1]`` first so they
    contribute on a comparable scale regardless of where they came from.
    The output is clamped to ``[0, 1]`` to keep PNG quantisation clean.
    """
    if bulk_depth.shape != micro_relief.shape:
        raise ValueError(
            f"shape mismatch: bulk {bulk_depth.shape} vs micro "
            f"{micro_relief.shape}"
        )
    total = float(bulk_weight) + float(micro_weight)
    if total <= 0.0:
        raise ValueError(
            f"weights must sum to a positive value; got {bulk_weight}+{micro_weight}"
        )
    wb = float(bulk_weight) / total
    wm = float(micro_weight) / total
    out = wb * normalise_unit(bulk_depth) + wm * normalise_unit(micro_relief)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
