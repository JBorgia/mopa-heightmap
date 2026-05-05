"""GET /blob/{id} — serve stored binary blobs with immutable caching."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..blob_store import load_bytes, get_meta

router = APIRouter(prefix="/blob", tags=["blob"])

_CACHE_HEADER = "public, max-age=31536000, immutable"


@router.get("/{blob_id}")
async def get_blob(blob_id: str) -> Response:
    data = load_bytes(blob_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Blob not found: {blob_id!r}")
    meta = get_meta(blob_id)
    content_type = meta[0] if meta else "application/octet-stream"
    return Response(
        content=data,
        media_type=content_type,
        headers={"Cache-Control": _CACHE_HEADER},
    )
