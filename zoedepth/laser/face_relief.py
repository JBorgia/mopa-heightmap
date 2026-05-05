"""Face-aware per-region depth weighting for portrait bas-relief.

This is what a sculptor would do by hand: deepen the nostrils, hollow the
eye sockets a touch, lift the nose tip and cheekbones, slightly recess the
chin underside, and amplify hair texture. Monocular depth networks produce
soft, even relief; this stage gives faces the punch the reference target
("sculptok") delivers, without any geometric hallucination.

The recipe is informed by Kerber-style face-relief and the conventions
documented in the Springer 2016 face-bas-relief paper. Magnitudes are
tuned so the default `face_relief_strength=1.0` is plausible on a stainless
or brass engraving budget; users dial it up or down per material.

Pipeline:
    1. Detect 468-landmark face mesh via MediaPipe FaceMesh.
    2. Compose soft per-region masks (Gaussian splats anchored to specific
       landmark indices) at the heightmap's resolution.
    3. Apply per-region additive offsets to the heightmap, scaled by the
       global strength knob and with sign flipped for `black_is_deep=False`.

If MediaPipe isn't installed or no face is detected, this module is a
no-op — engraving never fails because face-detection failed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "detect_face_landmarks",
    "build_region_masks",
    "apply_face_relief",
    "DEFAULT_REGION_OFFSETS",
    "DEFAULT_FACE_RELIEF_STRENGTH",
    "REGION_KEY_LANDMARKS",
]


# Default heightmap-unit offsets per region, tuned for a 0..1 heightmap with
# black_is_deep=True (1.0 = surface, 0.0 = deepest). Negative = deepen
# engraving; positive = raise (less engraving). These are "what a sculptor
# would push" — the values are intentionally subtle so a 1.0 strength is the
# baseline and the user can dial up to 1.5 for more aggressive sculpting.
DEFAULT_REGION_OFFSETS: Dict[str, float] = {
    "nostril_left":   -0.18,
    "nostril_right":  -0.18,
    "eye_socket_left":  -0.10,
    "eye_socket_right": -0.10,
    "iris_left":   -0.05,
    "iris_right":  -0.05,
    "nose_tip":     +0.07,
    "nose_bridge":  +0.05,
    "cheekbone_left":  +0.04,
    "cheekbone_right": +0.04,
    "lip_upper": +0.03,
    "lip_lower": +0.03,
    "chin_underside": -0.04,
    "brow_left":  +0.02,
    "brow_right": +0.02,
}


# A single global multiplier applied on top of the per-region offsets.
# 0.0 = stage is disabled; 1.0 = published values; up to 1.5 for hero shots.
DEFAULT_FACE_RELIEF_STRENGTH: float = 1.0


# Specific MediaPipe FaceMesh landmark indices used as splat centers. These
# are stable across MediaPipe versions; double-checked against
# mediapipe.solutions.face_mesh_connections constants.
REGION_KEY_LANDMARKS: Dict[str, Tuple[int, ...]] = {
    "nostril_left":   (327, 294, 392),
    "nostril_right":  (98, 64, 166),
    "eye_socket_left":  (362, 263, 386, 374),
    "eye_socket_right": (33, 133, 159, 145),
    "iris_left":   (473,),
    "iris_right":  (468,),
    "nose_tip":     (1, 4, 5),
    "nose_bridge":  (168, 6, 197, 195),
    "cheekbone_left":  (454, 366, 401),
    "cheekbone_right": (234, 137, 177),
    "lip_upper": (0, 11, 12, 13),
    "lip_lower": (14, 15, 16, 17),
    "chin_underside": (152, 175, 199),
    "brow_left":  (336, 296, 334, 293),
    "brow_right": (107, 66, 105, 63),
}


# Per-region splat radii in face-width units (face width = horizontal span
# of FACEMESH_FACE_OVAL). 0.05 ≈ 5 % of face width. Each region's effective
# Gaussian σ is `radius_units * face_width_px / 3` so the splat tapers off
# inside the documented region.
REGION_RADIUS_UNITS: Dict[str, float] = {
    "nostril_left":   0.025,
    "nostril_right":  0.025,
    "eye_socket_left":  0.060,
    "eye_socket_right": 0.060,
    "iris_left":   0.030,
    "iris_right":  0.030,
    "nose_tip":     0.040,
    "nose_bridge":  0.050,
    "cheekbone_left":  0.080,
    "cheekbone_right": 0.080,
    "lip_upper": 0.040,
    "lip_lower": 0.040,
    "chin_underside": 0.060,
    "brow_left":  0.050,
    "brow_right": 0.050,
}


@dataclass(frozen=True)
class FaceLandmarks:
    """468 normalised (x, y) landmarks, plus the integer image dims."""

    points: np.ndarray   # (468, 2) float32 in pixel coords
    width: int
    height: int

    @property
    def face_width_px(self) -> float:
        # Use cheek-to-cheek span (landmarks 234 ↔ 454) as the canonical
        # face-width measure — robust to head rotation in roll, and the
        # reference for our radius-in-face-width units.
        left = self.points[234]
        right = self.points[454]
        return float(np.linalg.norm(right - left)) or 1.0


# URL of the face_landmarker.task bundle (officially hosted by Google).
# Pinned float16 v1 — same model the legacy ``solutions.face_mesh`` API
# wrapped, exposing 478 landmarks (468 face mesh + 10 iris) with attention.
_FACE_LANDMARKER_TASK_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_landmarker_task() -> str:
    """Download the face_landmarker.task model on first use; cache locally."""
    import urllib.request
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "mopa-heightmap" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "face_landmarker.task"
    if not target.exists():
        urllib.request.urlretrieve(_FACE_LANDMARKER_TASK_URL, target)
    return str(target)


# MediaPipe BlazeFace was trained on selfie-style images where faces span
# 256-512 px. Below ~150 px reliable detection drops off; above ~1500 px
# the detector can also miss because the face takes up too much of the
# frame. We auto-upscale tiny inputs and downscale huge ones into the
# sweet spot for detection only — the landmarks come back rescaled to the
# original frame.
_DETECT_MIN_LONG_SIDE: int = 1024
_DETECT_MAX_LONG_SIDE: int = 1536


# Outer-corner landmark indices for the left and right eyes — used by
# :func:`auto_orient_to_face` to compute the inter-pupillary roll angle.
_LEFT_EYE_OUTER_LANDMARK: int = 263
_RIGHT_EYE_OUTER_LANDMARK: int = 33


def auto_orient_to_face(
    image: Image.Image,
    *,
    max_rotation_deg: float = 30.0,
    min_rotation_deg: float = 1.5,
) -> Image.Image:
    """Rotate ``image`` so the inter-pupillary line is horizontal.

    Uses the same MediaPipe FaceMesh detector as :mod:`face_relief`. If no
    face is found, or the corrected angle is below ``min_rotation_deg``,
    or above ``max_rotation_deg`` (suspect detection), the image is
    returned unchanged. Background fill on the rotated canvas is white
    (255), matching the engraving "no carve" plane convention.
    """
    landmarks = detect_face_landmarks(image)
    if landmarks is None:
        return image
    if (_LEFT_EYE_OUTER_LANDMARK >= len(landmarks.points)
            or _RIGHT_EYE_OUTER_LANDMARK >= len(landmarks.points)):
        return image
    left = landmarks.points[_LEFT_EYE_OUTER_LANDMARK]
    right = landmarks.points[_RIGHT_EYE_OUTER_LANDMARK]
    dx = float(left[0] - right[0])
    dy = float(left[1] - right[1])
    if dx == 0 and dy == 0:
        return image
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    if abs(angle_deg) < float(min_rotation_deg):
        return image
    if abs(angle_deg) > float(max_rotation_deg):
        return image
    # PIL.Image.rotate uses positive=counter-clockwise; our angle is the
    # tilt of the inter-pupillary line and we want to UNDO that tilt, so
    # rotate by the same sign.
    return image.rotate(
        angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255),
    )


def detect_face_landmarks(
    image: Image.Image,
    *,
    min_face_detection_confidence: float = 0.3,
) -> FaceLandmarks | None:
    """Return 478-point face mesh for the largest face in ``image``.

    Returns ``None`` when MediaPipe isn't installed, when no face is
    detected, or when the input is too small (< 64 px on the short side).
    Uses the modern MediaPipe Tasks API
    (``mediapipe.tasks.vision.FaceLandmarker``); the model bundle is fetched
    on first call and cached under ``~/.cache/mopa-heightmap/mediapipe``.

    Detection runs on a resized copy in a size range BlazeFace likes
    (~1024 px on the long side). Returned landmarks are in the original
    image's coordinate space.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if min(w, h) < 64:
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    try:
        task_path = _ensure_face_landmarker_task()
    except Exception:
        return None

    long_side = max(w, h)
    if long_side < _DETECT_MIN_LONG_SIDE:
        scale = _DETECT_MIN_LONG_SIDE / float(long_side)
    elif long_side > _DETECT_MAX_LONG_SIDE:
        scale = _DETECT_MAX_LONG_SIDE / float(long_side)
    else:
        scale = 1.0
    if scale != 1.0:
        det_w = max(1, int(round(w * scale)))
        det_h = max(1, int(round(h * scale)))
        det_image = rgb.resize((det_w, det_h), Image.BICUBIC)
    else:
        det_image = rgb

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=task_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=float(min_face_detection_confidence),
        min_face_presence_confidence=float(min_face_detection_confidence),
        min_tracking_confidence=float(min_face_detection_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    arr = np.asarray(det_image, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    # Landmarks are normalised [0, 1] coords in the detection-image frame.
    # Multiplying by the *original* w/h maps back to source pixel coords.
    pts = np.array(
        [(lm.x * w, lm.y * h) for lm in landmarks],
        dtype=np.float32,
    )
    return FaceLandmarks(points=pts, width=w, height=h)


def _gaussian_splat(
    h: int,
    w: int,
    cx: float,
    cy: float,
    sigma: float,
) -> np.ndarray:
    """Single-point Gaussian splat. Returns ``(h, w)`` float32 in [0, 1]."""
    sigma = max(float(sigma), 1.0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def build_region_masks(
    landmarks: FaceLandmarks,
    shape: Tuple[int, int],
    *,
    radius_units: Dict[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    """Build a soft mask per face region anchored on the given landmarks.

    Each mask is a sum of Gaussian splats at ``REGION_KEY_LANDMARKS[name]``,
    σ scaled by ``radius_units[name] * face_width_px / 3``. Output masks are
    normalised so the peak is 1.0; downstream code multiplies by the
    region-specific signed offset.
    """
    if radius_units is None:
        radius_units = REGION_RADIUS_UNITS
    h, w = shape
    fw = landmarks.face_width_px
    masks: Dict[str, np.ndarray] = {}
    for name, indices in REGION_KEY_LANDMARKS.items():
        radius = radius_units.get(name, 0.05)
        sigma = max(2.0, (radius * fw) / 3.0)
        accum = np.zeros((h, w), dtype=np.float32)
        for idx in indices:
            if idx >= len(landmarks.points):
                continue
            cx, cy = landmarks.points[idx]
            accum += _gaussian_splat(h, w, cx, cy, sigma)
        peak = float(accum.max())
        if peak > 1e-6:
            accum = accum / peak
        masks[name] = accum
    return masks


def apply_face_relief(
    heightmap: np.ndarray,
    image: Image.Image,
    *,
    strength: float = DEFAULT_FACE_RELIEF_STRENGTH,
    black_is_deep: bool = True,
    region_offsets: Dict[str, float] | None = None,
    landmarks: FaceLandmarks | None = None,
) -> np.ndarray:
    """Apply per-region depth offsets to ``heightmap`` for a detected face.

    Returns the heightmap unchanged when no face is detected. The output
    is clipped to ``[0, 1]`` so downstream PNG quantisation stays clean.
    """
    if strength <= 0.0:
        return heightmap.astype(np.float32, copy=False)

    if landmarks is None:
        landmarks = detect_face_landmarks(image)
    if landmarks is None:
        return heightmap.astype(np.float32, copy=False)

    if region_offsets is None:
        region_offsets = DEFAULT_REGION_OFFSETS

    masks = build_region_masks(landmarks, heightmap.shape)
    # Region offsets are written in "raise = positive, deepen = negative"
    # semantics on the standard `black_is_deep=True` heightmap (1=surface).
    # When black_is_deep is False the polarity flips, so flip the sign.
    polarity = 1.0 if black_is_deep else -1.0

    out = heightmap.astype(np.float32, copy=True)
    for name, offset in region_offsets.items():
        mask = masks.get(name)
        if mask is None:
            continue
        out = out + (polarity * float(offset) * float(strength)) * mask
    return np.clip(out, 0.0, 1.0).astype(np.float32)
