"""Sculptok auto-pull endpoints.

Two routes:

    GET  /sculptok/credits  — configured status + remaining credit balance.
    POST /sculptok/generate — generate a heightmap from a previously-uploaded
                              image_id, return the on-disk path the render
                              endpoint reads via settings.external_heightmap_path.

The API key is resolved server-side via mopa.settings.resolve_sculptok_api_key
(same lookup chain the CLI uses: ~/.mopa-heightmap/settings.json or
SCULPTOK_API_KEY env). If unconfigured, /credits reports it gracefully and
/generate returns 503.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from mopa.sculptok_client import (
    SculptokAPIError,
    SculptokClient,
    SculptokDepthMapParams,
    SculptokInsufficientCreditsError,
)
from mopa.settings import load_settings, resolve_sculptok_api_key

from ..schemas import (
    SculptokCreditsResponse,
    SculptokGenerateRequest,
    SculptokGenerateResponse,
)
from ..service_adapter import get_upload


router = APIRouter(prefix="/sculptok", tags=["sculptok"])


# Server-side cache directory for sculptok-generated heightmaps. Persisted
# across requests so /render can read what /generate wrote.
_CACHE_DIR = Path(tempfile.gettempdir()) / "mopa_sculptok_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_client() -> SculptokClient | None:
    """Return a Sculptok client when an API key is configured, else None."""
    api_key = resolve_sculptok_api_key(cli_value=None, settings=load_settings())
    if not api_key:
        return None
    return SculptokClient(api_key)


@router.get("/credits", response_model=SculptokCreditsResponse)
async def credits() -> SculptokCreditsResponse:
    client = _resolve_client()
    if client is None:
        return SculptokCreditsResponse(configured=False, balance=None)
    try:
        balance = client.get_credits()
    except SculptokAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Sculptok API error: {exc}") from exc
    return SculptokCreditsResponse(configured=True, balance=int(balance))


@router.post("/generate", response_model=SculptokGenerateResponse)
async def generate(req: SculptokGenerateRequest) -> SculptokGenerateResponse:
    image = get_upload(req.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {req.image_id}")

    client = _resolve_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sculptok API key not configured on the server. Set "
                "SCULPTOK_API_KEY env or add credentials.sculptok_api_key "
                "to ~/.mopa-heightmap/settings.json."
            ),
        )

    params = SculptokDepthMapParams(
        style=req.style, version=req.version, draw_hd=req.draw_hd,
    )

    # Persist the photo to a temp file for the upload step (the client's
    # one-shot generate_heightmap() expects an on-disk source).
    photo_path = _CACHE_DIR / f"{req.image_id}_photo.png"
    image.save(photo_path, format="PNG")
    out_path = _CACHE_DIR / f"{req.image_id}_sculptok.png"

    balance_before: int
    try:
        balance_before = client.get_credits()
    except SculptokAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Sculptok API error: {exc}") from exc

    try:
        result_path = client.generate_heightmap(
            photo_path,
            params=params,
            out_path=out_path,
            check_credits=False,  # we just checked above
        )
    except SculptokInsufficientCreditsError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except SculptokAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Sculptok API error: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    try:
        balance_after = client.get_credits()
    except SculptokAPIError:
        balance_after = max(0, balance_before - params.expected_cost())

    return SculptokGenerateResponse(
        heightmap_path=str(result_path.resolve()),
        credits_used=max(0, balance_before - balance_after),
        credits_remaining=int(balance_after),
    )
