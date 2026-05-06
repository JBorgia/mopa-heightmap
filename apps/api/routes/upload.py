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

from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ..schemas import UploadResponse
from ..service_adapter import store_upload

router = APIRouter(prefix="/upload", tags=["upload"])


# Same cache dir the sculptok auto-pull route writes to so /render
# reads either source uniformly.
_HEIGHTMAP_CACHE_DIR = Path(tempfile.gettempdir()) / "mopa_sculptok_cache"
_HEIGHTMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    """

    heightmap_path: str
    width: int
    height: int


@router.post("/heightmap", response_model=HeightmapUploadResponse)
async def upload_heightmap(file: UploadFile) -> HeightmapUploadResponse:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode heightmap: {exc}") from exc

    suffix = Path(file.filename or "heightmap.png").suffix.lower() or ".png"
    if suffix not in {".png", ".tif", ".tiff", ".bmp"}:
        suffix = ".png"
    out = _HEIGHTMAP_CACHE_DIR / f"upload_{uuid.uuid4().hex[:12]}{suffix}"
    image.save(out)
    return HeightmapUploadResponse(
        heightmap_path=str(out.resolve()),
        width=image.width,
        height=image.height,
    )
