"""POST /upload — receive a multipart image, store it, return metadata."""
from __future__ import annotations

import io

from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..schemas import UploadResponse
from ..service_adapter import store_upload

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_image(file: UploadFile) -> UploadResponse:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc
    return store_upload(image)
