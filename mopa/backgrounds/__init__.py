"""Procedural background-pattern generators for the pre-sculptok BG-replace flow.

Each generator returns a single-channel float32 array in ``[0, 1]``
shaped ``(H, W)``. The expected use is:

    1. Subject mask the original photo (BiRefNet / rembg / threshold)
       to know where the background pixels are.
    2. Render a procedural background at the photo's dimensions.
    3. Composite: ``photo[bg_mask] = pattern[bg_mask]``.
    4. Send the composited photo to sculptok. Sculptok now sees a
       full-frame relief signal (subject + decorative background)
       instead of a flat background that's hard to read.

Patterns ship: ``guilloche``, ``stripes``, ``dots``, ``halftone``,
``checkers``. Each accepts a common ``(width, height, scale, angle,
seed)`` kwargs subset; pattern-specific tuning lives on the individual
function.
"""
from __future__ import annotations

from .generators import (
    PATTERN_NAMES,
    checkers_pattern,
    dots_pattern,
    generate_pattern,
    guilloche_pattern,
    halftone_pattern,
    stripes_pattern,
)

__all__ = [
    "PATTERN_NAMES",
    "checkers_pattern",
    "dots_pattern",
    "generate_pattern",
    "guilloche_pattern",
    "halftone_pattern",
    "stripes_pattern",
]
