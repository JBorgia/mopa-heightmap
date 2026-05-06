"""Auto-orient: rotate the photo so the inter-pupillary line is level.

Sculptok is sensitive to subject orientation — a 30° head tilt often
produces a noticeably worse depth output than the same head upright.
This module finds eyes via OpenCV's Haar cascade and rotates the image
so the line through the two eye centres is horizontal.

OpenCV Haar is the right detector here because it ships with
``opencv-contrib-python`` (already a project dependency) and needs no
model-file download. MediaPipe's FaceLandmarker is more accurate but
requires a 5 MB ``.task`` blob the user has to fetch on first use; not
worth the friction for the modest gain on this pipeline.

No-ops when:
  * No face is detected.
  * Fewer than 2 eyes detected on the face.
  * Detected face is small (< 5 % of the longest side).
  * The required rotation is below ``min_angle_deg`` (default 1°).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image


__all__ = ["auto_orient_to_face", "find_face_eye_angle"]


def _cascade_path(name: str) -> str:
    import cv2
    return os.path.join(cv2.data.haarcascades, name)


def find_face_eye_angle(
    image: Image.Image,
    *,
    min_face_fraction: float = 0.05,
) -> Optional[float]:
    """Return the rotation angle (degrees) that levels the eyes, or None.

    Positive angle = clockwise rotation needed. Returns None when no
    face / not enough eyes / face too small.
    """
    try:
        import cv2
    except ImportError:
        return None

    arr = np.asarray(image.convert("L"))
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest < 32:
        return None

    face_cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_eye.xml"))
    if face_cascade.empty() or eye_cascade.empty():
        return None

    faces = face_cascade.detectMultiScale(
        arr, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    )
    if len(faces) == 0:
        return None
    # Pick the largest face.
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    fx, fy, fw, fh = faces[0]
    if fw / longest < float(min_face_fraction):
        return None

    face_roi = arr[fy:fy + fh, fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(
        face_roi, scaleFactor=1.1, minNeighbors=5,
        minSize=(max(8, fw // 12), max(8, fh // 12)),
    )
    if len(eyes) < 2:
        return None
    # Take the two largest eye boxes — keeps stray eyebrow/glasses
    # detections from biasing the angle.
    eyes = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]
    e1, e2 = eyes
    cx1 = fx + e1[0] + e1[2] / 2.0
    cy1 = fy + e1[1] + e1[3] / 2.0
    cx2 = fx + e2[0] + e2[2] / 2.0
    cy2 = fy + e2[1] + e2[3] / 2.0
    # Order by x so the angle sign matches "clockwise rotation needed".
    if cx2 < cx1:
        cx1, cx2 = cx2, cx1
        cy1, cy2 = cy2, cy1
    dx = cx2 - cx1
    dy = cy2 - cy1
    if abs(dx) < 1e-6:
        return None
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return angle


def auto_orient_to_face(
    image: Image.Image,
    *,
    min_angle_deg: float = 1.0,
) -> Tuple[Image.Image, float]:
    """Rotate so the eye-line is horizontal. Returns ``(rotated, angle)``.

    ``angle`` is 0 when no rotation is applied (no face / tiny face /
    below threshold / opencv unavailable).
    """
    angle = find_face_eye_angle(image)
    if angle is None or abs(angle) < float(min_angle_deg):
        return image, 0.0
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    return rotated, float(angle)
