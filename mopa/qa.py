"""Heuristic QA pass over a finished heightmap.

Cheap, no-model checks that catch the failure modes the research agent
flagged for monocular depth → laser engraving (background floater,
hollow cheeks, mirror-flipped depth, specular-pit, neck-shoulder seam).
Each check returns a list of :class:`QAFinding` so the CLI can warn
before the user starts a 90-minute laser job, and the API can pipe the
list into a "warnings" widget on the wizard.

Design rules:
    * Every check is a pure function with a default-empty return so QA
      reports never break a render.
    * Severity is one of ``"info" | "warning" | "error"``. ``error``
      means "do not engrave" (very rare); ``warning`` means "open the
      preview before you burn"; ``info`` is FYI only.
    * No external dependencies. All numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


__all__ = [
    "QAFinding",
    "qa_report",
    "check_background_floater",
    "check_dynamic_range",
    "check_mirror_asymmetry",
    "check_specular_pits",
    "check_floating_hair",
    "DEFAULT_BG_FLOATER_STD",
    "DEFAULT_MIN_DYNAMIC_RANGE",
    "DEFAULT_MIRROR_ASYMMETRY_THRESHOLD",
]


# A background that's truly flat (mask hard-flattened it to a plane) has
# std ≈ 0. Anything above a few percent of dynamic range means the mask
# leaked or the bg wasn't flattened. 1.5% catches real floaters without
# tripping on dither noise.
DEFAULT_BG_FLOATER_STD: float = 0.015

# Minimum heightmap span before we warn the user that the engraving will
# barely register on the material. Below this they're carving 1-2 µm
# differences which most metals can't resolve.
DEFAULT_MIN_DYNAMIC_RANGE: float = 0.15

# Threshold for left-right mean-depth difference. Lighting bias on a
# real face produces ≤ 0.05; > 0.10 strongly suggests the depth model
# emitted a mirror-flipped output (rare in DAv2, common in older
# Marigold checkpoints).
DEFAULT_MIRROR_ASYMMETRY_THRESHOLD: float = 0.10


@dataclass(frozen=True)
class QAFinding:
    """One QA observation. ``code`` is a stable string for I18n / tests."""

    code: str
    severity: str        # "info" | "warning" | "error"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


# --------------------------------------------------------------- checks

def check_background_floater(
    heightmap: np.ndarray,
    *,
    background_value: float = 1.0,
    threshold_std: float = DEFAULT_BG_FLOATER_STD,
) -> List[QAFinding]:
    """Background pixels (≈ ``background_value``) should be near-zero std."""
    bg = heightmap[np.abs(heightmap - background_value) < 0.02]
    if bg.size == 0:
        return []
    std = float(np.std(bg))
    if std > threshold_std:
        return [QAFinding(
            code="bg_floater",
            severity="warning",
            message=(
                f"Background pixels have std={std:.3f} (> {threshold_std:.3f}); "
                "subject mask may be leaking — enable subject_mask_enabled or "
                "check the BiRefNet alpha."
            ),
        )]
    return []


def check_dynamic_range(
    heightmap: np.ndarray,
    *,
    threshold: float = DEFAULT_MIN_DYNAMIC_RANGE,
) -> List[QAFinding]:
    """Subject relief amplitude must be large enough to actually engrave."""
    finite = heightmap[np.isfinite(heightmap)]
    if finite.size == 0:
        return [QAFinding(
            code="empty_heightmap",
            severity="error",
            message="Heightmap contains no finite values.",
        )]
    p5, p95 = np.percentile(finite, [5, 95])
    span = float(p95 - p5)
    if span < threshold:
        return [QAFinding(
            code="low_dynamic_range",
            severity="warning",
            message=(
                f"Heightmap p5..p95 span is only {span:.3f}; the engraver "
                "will barely register relief. Lower deep_limit or raise "
                "surface_limit, or pick a higher-contrast image."
            ),
        )]
    return []


def check_mirror_asymmetry(
    heightmap: np.ndarray,
    *,
    background_value: float = 1.0,
    threshold: float = DEFAULT_MIRROR_ASYMMETRY_THRESHOLD,
) -> List[QAFinding]:
    """Detect left/right mean-depth imbalance that suggests a flipped depth pred."""
    h, w = heightmap.shape
    if w < 4:
        return []
    is_subject = np.abs(heightmap - background_value) >= 0.02
    left = heightmap[:, : w // 2][is_subject[:, : w // 2]]
    right = heightmap[:, w // 2 :][is_subject[:, w // 2 :]]
    if left.size < 100 or right.size < 100:
        return []
    delta = float(abs(np.mean(left) - np.mean(right)))
    if delta > threshold:
        return [QAFinding(
            code="mirror_asymmetry",
            severity="info",
            message=(
                f"Left/right mean depth differs by {delta:.3f}; "
                "mostly cosmetic — but if the photo is symmetric this can "
                "indicate a mirror-flipped depth prediction."
            ),
        )]
    return []


def check_specular_pits(
    heightmap: np.ndarray,
    photo: Optional[np.ndarray],
    *,
    background_value: float = 1.0,
    specular_luma: float = 0.97,
    pit_depth: float = 0.5,
) -> List[QAFinding]:
    """Bright photo highlights that read as deep pits — common on jewelry / glasses."""
    if photo is None:
        return []
    if photo.shape[:2] != heightmap.shape:
        return []
    luma = (
        photo if photo.ndim == 2 else
        0.2126 * photo[..., 0] + 0.7152 * photo[..., 1] + 0.0722 * photo[..., 2]
    )
    luma = np.asarray(luma, dtype=np.float32)
    if luma.max() > 1.5:
        luma = luma / 255.0
    bright = luma > float(specular_luma)
    pit = heightmap < float(pit_depth)
    pixels = int(np.sum(bright & pit))
    if pixels > 50:                # ignore pinprick highlights
        return [QAFinding(
            code="specular_as_pit",
            severity="warning",
            message=(
                f"{pixels} pixels are bright in the photo but read as deep pits "
                "(eyes, jewelry, lipstick). Enable Marigold-IID delighting "
                "before depth estimation to suppress."
            ),
        )]
    return []


def check_floating_hair(
    heightmap: np.ndarray,
    *,
    top_strip_fraction: float = 0.15,
    background_value: float = 1.0,
    floor_threshold: float = 0.6,
) -> List[QAFinding]:
    """Hair near the top of the frame that floats above the head silhouette.

    A common DA-V2 failure: hair pixels are read as *closer* than the face,
    so they end up brighter (more raised) than the head — which laser-
    engraves as a flat plate sitting above everything else. Heuristic
    checks the top strip and flags if its mean depth is *higher* (closer
    to surface) than the rest of the subject by a wide margin.
    """
    h, w = heightmap.shape
    strip_h = max(1, int(h * top_strip_fraction))
    is_subject = np.abs(heightmap - background_value) >= 0.02
    top = heightmap[:strip_h][is_subject[:strip_h]]
    rest = heightmap[strip_h:][is_subject[strip_h:]]
    if top.size < 50 or rest.size < 100:
        return []
    if float(np.mean(top)) > float(np.mean(rest)) + (1.0 - floor_threshold):
        return [QAFinding(
            code="floating_hair",
            severity="info",
            message=(
                "Top of subject reads as raised above the rest — check for "
                "floating-hair artefact; consider lowering relief_strength "
                "or enabling face_relief."
            ),
        )]
    return []


# --------------------------------------------------------------- aggregator

def qa_report(
    heightmap: np.ndarray,
    *,
    photo: Optional[np.ndarray] = None,
    background_value: float = 1.0,
) -> List[QAFinding]:
    """Run every available check and return the concatenated finding list."""
    findings: List[QAFinding] = []
    findings.extend(check_background_floater(heightmap, background_value=background_value))
    findings.extend(check_dynamic_range(heightmap))
    findings.extend(check_mirror_asymmetry(heightmap, background_value=background_value))
    findings.extend(check_specular_pits(heightmap, photo, background_value=background_value))
    findings.extend(check_floating_hair(heightmap, background_value=background_value))
    return findings
