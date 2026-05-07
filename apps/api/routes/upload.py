"""POST /upload — receive a multipart image, store it, return metadata.

Two endpoints share this router:
  * ``POST /upload``           — the source photo. In-memory store keyed
                                  by sha256, returned as ``image_id``.
  * ``POST /upload/heightmap`` — a sculptok / meshy / hand-authored
                                  heightmap PNG. Persisted to the same
                                  temp dir the sculptok auto-pull route
                                  uses, returned as a server-side path
                                  the client drops into
                                  ``settings.external_heightmap_path``.
"""
from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .. import blob_store
from ..schemas import UploadResponse
from ..service_adapter import store_upload

router = APIRouter(prefix="/upload", tags=["upload"])


# Same cache dir the sculptok auto-pull route writes to so /render
# reads either source uniformly.
_HEIGHTMAP_CACHE_DIR = Path(tempfile.gettempdir()) / "mopa_sculptok_cache"
_HEIGHTMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Tuned against Sculptok's web-UI side-by-side composite: the depth map
# (white relief on black) sits beside a render preview (white relief on
# mid-grey). The two halves contain the *same silhouette* with a
# different background; a real wide heightmap has its subject split
# across the seam (different content in each half).
#
# Detection: threshold each half to a binary subject mask, centroid-align
# the right half onto the left, and compute IoU. High IoU + similar
# coverage ⇒ same shape twice ⇒ composite. We then crop to whichever
# half has the darker mean (that's the depth map; the preview is
# brighter overall because of the grey ground).
_COMPOSITE_MIN_ASPECT_RATIO = 1.4
# Real Sculptok composites have a TRUE-BLACK background with the relief
# silhouette in white (depth map) or grey (preview) on top. Pixels above
# this threshold are "non-background" — the silhouette. Sculptok previews
# render on a black canvas with a smaller grey viewport around the
# subject, so the dominant background of both halves is still 0.
_COMPOSITE_SUBJECT_THRESHOLD = 30
_COMPOSITE_MIN_SUBJECT_PIXELS = 100     # need a real subject in each half
_COMPOSITE_IOU_MIN = 0.6                # aligned silhouettes must agree
_COMPOSITE_COVERAGE_RATIO_MIN = 0.7     # halves must use a similar amount of subject
# Composites pair a depth-map render with a different shading style
# (preview), so the two halves have measurably different mean luminance.
# A symmetric subject split across the seam has near-identical halves
# and would otherwise fool the IoU check — this gates that case out.
_COMPOSITE_MEAN_DIFF_MIN = 5.0


def _shifted_overlay(target_shape: Tuple[int, int], src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Place ``src`` into a ``target_shape`` canvas at offset (dy, dx).
    Pixels that fall outside the canvas are dropped silently."""
    out = np.zeros(target_shape, dtype=src.dtype)
    sh, sw = src.shape
    th, tw = target_shape
    sy0, sy1 = max(0, -dy), sh - max(0, dy + sh - th)
    sx0, sx1 = max(0, -dx), sw - max(0, dx + sw - tw)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    dy0, dx0 = max(0, dy), max(0, dx)
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = src[sy0:sy1, sx0:sx1]
    return out


def _detect_and_crop_composite(image: Image.Image) -> Tuple[Image.Image, bool]:
    """If ``image`` looks like a Sculptok side-by-side composite (depth map
    next to a render preview of the same relief), return the depth-map
    half and True. Otherwise return the original image and False.
    """
    w, h = image.size
    if w == 0 or h == 0 or (w / h) < _COMPOSITE_MIN_ASPECT_RATIO:
        return image, False

    arr = np.asarray(image.convert("L"))
    mid = w // 2
    left_arr = arr[:, :mid]
    right_arr = arr[:, mid:]

    # Crop both halves to the same width — odd-width inputs produce a
    # right half one pixel wider than the left, which breaks np boolean
    # ops below.
    half_w = min(left_arr.shape[1], right_arr.shape[1])
    left_arr = left_arr[:, :half_w]
    right_arr = right_arr[:, :half_w]

    # The two halves of a composite have visibly different shading (depth
    # map vs render preview) — the mean luminance differs. A symmetric
    # subject split across the seam (e.g. a centred portrait) would have
    # nearly identical halves and would otherwise pass the IoU check.
    left_mean = float(left_arr.mean())
    right_mean = float(right_arr.mean())
    if abs(left_mean - right_mean) < _COMPOSITE_MEAN_DIFF_MIN:
        return image, False

    L_mask = left_arr > _COMPOSITE_SUBJECT_THRESHOLD
    R_mask = right_arr > _COMPOSITE_SUBJECT_THRESHOLD
    if L_mask.sum() < _COMPOSITE_MIN_SUBJECT_PIXELS or R_mask.sum() < _COMPOSITE_MIN_SUBJECT_PIXELS:
        return image, False

    # Coverage similarity: a composite shows the same subject with the
    # same area; a landscape heightmap split in half doesn't.
    cov_l = float(L_mask.mean())
    cov_r = float(R_mask.mean())
    cov_ratio = min(cov_l, cov_r) / max(cov_l, cov_r) if max(cov_l, cov_r) else 0.0
    if cov_ratio < _COMPOSITE_COVERAGE_RATIO_MIN:
        return image, False

    # Centroid-align R onto L's frame, then compute IoU.
    ys_l, xs_l = np.where(L_mask)
    ys_r, xs_r = np.where(R_mask)
    dy = int(round(float(ys_l.mean()) - float(ys_r.mean())))
    dx = int(round(float(xs_l.mean()) - float(xs_r.mean())))
    R_aligned = _shifted_overlay(L_mask.shape, R_mask.astype(np.uint8), dy, dx).astype(bool)
    inter = np.logical_and(L_mask, R_aligned).sum()
    union = np.logical_or(L_mask, R_aligned).sum()
    iou = float(inter / union) if union else 0.0
    if iou < _COMPOSITE_IOU_MIN:
        return image, False

    # Sculptok renders the depth map on black; the preview sits on grey.
    # Crop to the half with the darker mean — that's the depth data.
    keep_left = left_mean <= right_mean
    box = (0, 0, mid, h) if keep_left else (mid, 0, w, h)
    return image.crop(box), True


@router.post("", response_model=UploadResponse)
async def upload_image(file: UploadFile) -> UploadResponse:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc
    return store_upload(image)


class HeightmapUploadResponse(BaseModel):
    """Shape of POST /upload/heightmap. The path goes straight into
    ``settings.external_heightmap_path`` for the next /render call.

    ``auto_cropped`` flags the case where we detected a Sculptok-style
    side-by-side composite (depth map + render preview) and saved only
    the depth-map half. The wizard surfaces this as a toast so the user
    knows their PNG was modified.
    """

    heightmap_path: str
    width: int
    height: int
    auto_cropped: bool = False


@router.post("/heightmap", response_model=HeightmapUploadResponse)
async def upload_heightmap(file: UploadFile) -> HeightmapUploadResponse:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode heightmap: {exc}") from exc

    image, auto_cropped = _detect_and_crop_composite(image)

    suffix = Path(file.filename or "heightmap.png").suffix.lower() or ".png"
    if suffix not in {".png", ".tif", ".tiff", ".bmp"}:
        suffix = ".png"
    out = _HEIGHTMAP_CACHE_DIR / f"upload_{uuid.uuid4().hex[:12]}{suffix}"
    image.save(out)
    return HeightmapUploadResponse(
        heightmap_path=str(out.resolve()),
        width=image.width,
        height=image.height,
        auto_cropped=auto_cropped,
    )
