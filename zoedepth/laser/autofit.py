"""Automatic tone-curve fitting from a raw depth histogram.

Given a ZoeDepth metric-depth map (larger value = farther), pick sensible
``near_percentile``, ``far_percentile``, ``gamma``, ``deep_limit``, and
``surface_limit`` so the subject occupies most of the dynamic range and a
dominant far-plane background gets clipped to the deepest engraving level.

The pipeline downstream (heightmap.process_depth_to_heightmap) consumes these
keys directly, so the result is just dropped into the same overrides dict the
sliders already produce.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def autofit_overrides_from_depth(depth: np.ndarray) -> Dict[str, Any]:
    """Return suggested override values keyed by the same names as the sliders.

    Convention: ``black_is_deep=True`` (the project default). After
    normalize_depth + orient_for_lightburn, close pixels become bright and
    far pixels become dark. So clipping a far-plane background drives it to
    bright in normalized space, then dark (deepest engraving) after orient —
    which is what we want for raised-relief subjects on a flat background.
    """
    d = depth.astype(np.float32).ravel()
    d = d[np.isfinite(d)]
    if d.size < 100:
        return {}

    lo = float(np.percentile(d, 0.5))
    hi = float(np.percentile(d, 99.5))
    if hi - lo < 1e-6:
        return {}
    n = np.clip((d - lo) / (hi - lo), 0.0, 1.0)

    hist, edges = np.histogram(n, bins=128, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(hist.sum())
    if total <= 0:
        return {}

    # Detect dominant far-plane background mode in the last 25% of bins.
    far_start = int(0.75 * len(hist))
    far_peak = far_start + int(np.argmax(hist[far_start:]))
    far_peak_frac = hist[far_peak] / total
    has_background = far_peak_frac > 0.04

    bg_threshold_norm = 1.0
    if has_background:
        # Walk left from the far peak to the trough between subject and bg.
        trough = far_peak
        for i in range(far_peak - 1, 0, -1):
            if hist[i] <= hist[trough]:
                trough = i
            elif hist[i] > hist[trough] * 2.0 and hist[i] > total * 0.005:
                break
        bg_threshold_norm = float(centers[trough])

    subject = n[n < bg_threshold_norm]
    if subject.size < 100:
        return {
            "near_percentile": 5.0,
            "far_percentile": 95.0,
            "gamma": 0.72,
            "deep_limit": 0.02,
            "surface_limit": 0.98,
            "detail_mode": "highpass",
            "detail_strength": 0.10,
            "detail_highpass_radius": 9,
            "detail_subject_mask": True,
            "detail_invert": False,
        }

    # Map subject percentiles back to full-distribution percentiles via the
    # empirical CDF — that's exactly what the heightmap pipeline expects.
    sub_low = float(np.percentile(subject, 2.0))
    sub_high = float(np.percentile(subject, 98.0))
    near_pct = float((n <= sub_low).mean()) * 100.0
    far_pct = float((n <= sub_high).mean()) * 100.0
    near_pct = max(0.5, min(near_pct, 30.0))
    far_pct = max(60.0, min(far_pct, 99.5))
    if far_pct - near_pct < 5.0:
        far_pct = min(99.5, near_pct + 5.0)

    # Gamma: pick so the subject median lands at 0.5 in the bright-oriented
    # heightmap (after normalize+orient with black_is_deep=True).
    sub_med = float(np.median(subject))
    if sub_high > sub_low:
        m_clip = float(np.clip((sub_med - sub_low) / (sub_high - sub_low), 0.05, 0.95))
        m_oriented = 1.0 - m_clip  # black_is_deep=True
        if 0.05 < m_oriented < 0.95:
            gamma = float(np.log(0.5) / np.log(m_oriented))
            gamma = max(0.4, min(gamma, 1.8))
        else:
            gamma = 0.72
    else:
        gamma = 0.72

    return {
        "near_percentile": round(near_pct, 1),
        "far_percentile": round(far_pct, 1),
        "gamma": round(gamma, 2),
        "deep_limit": 0.02,
        "surface_limit": 0.98,
        # Stage B — conservative high-pass-only default.
        # Luminance and "both" modes treat photo color as depth, which is
        # non-physical and produces ink-blot artefacts; they remain available
        # but are no longer suggested by autofit. See IMPLEMENTATION_PLAN.md
        # §2 "Why we are rebuilding". Strength 0.10 mirrors ReliefGenerater's
        # α ≈ 0.05 (we add a touch since we high-pass instead of straight blend).
        "detail_mode": "highpass",
        "detail_strength": 0.10,
        "detail_highpass_radius": 9,
        "detail_subject_mask": True,
        "detail_invert": False,
    }
