"""Target-aware auto-crop.

Two centring strategies share a single entry point:

  * **Face-centred** (default for portrait targets): use OpenCV's Haar
    cascade to find the dominant face bounding box, then expand to
    ``target_aspect`` centred on it.
  * **Saliency-centred** (fallback): pick the centre-of-mass of
    ``cv2.saliency.StaticSaliencySpectralResidual`` — bright /
    high-contrast region.
  * **Centre crop** (last-resort fallback): when neither face nor
    saliency yields a useful centre, crop centred on the image.

All paths return a PIL image cropped to the requested aspect ratio.
The caller is responsible for any subsequent resize.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "auto_crop_to_aspect",
    "find_face_bbox",
    "find_saliency_centre",
]


def _cascade_path(name: str) -> str:
    import cv2
    return os.path.join(cv2.data.haarcascades, name)


def find_face_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` of the dominant face, or None."""
    try:
        import cv2
    except ImportError:
        return None
    arr = np.asarray(image.convert("L"))
    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        return None
    cascade = cv2.CascadeClassifier(_cascade_path("haarcascade_frontalface_default.xml"))
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(
        arr, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    )
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    return int(fx), int(fy), int(fx + fw), int(fy + fh)


def find_saliency_centre(image: Image.Image) -> Tuple[int, int]:
    """Return ``(cx, cy)`` of the most-salient region, or the image centre."""
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    try:
        import cv2
        saliency_module = getattr(cv2, "saliency", None)
        if saliency_module is None:
            raise AttributeError
        sal = saliency_module.StaticSaliencySpectralResidual_create()
        ok, smap = sal.computeSaliency(arr)
        if not ok:
            raise RuntimeError("saliency failed")
        smap = smap.astype(np.float32)
        if smap.sum() <= 0:
            raise RuntimeError("flat saliency")
        ys, xs = np.indices(smap.shape)
        cy = int(round((ys * smap).sum() / smap.sum()))
        cx = int(round((xs * smap).sum() / smap.sum()))
        return cx, cy
    except Exception:
        return w // 2, h // 2


def auto_crop_to_aspect(
    image: Image.Image,
    *,
    target_aspect: float,
    prefer_face: bool = True,
) -> Tuple[Image.Image, str]:
    """Crop ``image`` to ``target_aspect`` (width / height).

    Returns ``(cropped, strategy)`` where ``strategy`` is one of
    ``"face"``, ``"saliency"``, ``"center"``.
    """
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    if w <= 0 or h <= 0:
        return image, "center"

    cx: int
    cy: int
    strategy: str

    if prefer_face:
        bbox = find_face_bbox(image)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            strategy = "face"
        else:
            cx, cy = find_saliency_centre(image)
            strategy = "saliency" if (cx, cy) != (w // 2, h // 2) else "center"
    else:
        cx, cy = find_saliency_centre(image)
        strategy = "saliency" if (cx, cy) != (w // 2, h // 2) else "center"

    if target_aspect >= w / h:
        crop_w = w
        crop_h = int(round(crop_w / target_aspect))
        crop_h = min(crop_h, h)
    else:
        crop_h = h
        crop_w = int(round(crop_h * target_aspect))
        crop_w = min(crop_w, w)

    half_w = crop_w // 2
    half_h = crop_h // 2
    x0 = max(0, min(w - crop_w, cx - half_w))
    y0 = max(0, min(h - crop_h, cy - half_h))
    cropped = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return cropped, strategy
