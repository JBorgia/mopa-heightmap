"""Sculptok endpoints.

    GET  /sculptok/credits          — configured status + credit balance
    POST /sculptok/generate         — generate heightmap (async via ARQ when
                                      Redis is configured, synchronous fallback)
    GET  /sculptok/jobs/{job_id}    — poll async job status

When REDIS_URL is set the POST returns immediately with {job_id} and the
frontend polls /jobs/{job_id} until {status: "done"}.  When Redis is absent
the POST runs synchronously as before — local dev still works unchanged.

Credit enforcement: if SUPABASE_JWT_SECRET is configured, the generate route
requires a valid Bearer token and deducts one credit on success. Without the
JWT secret the endpoint is open (dev mode).
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from PIL import Image

from mopa.sculptok_client import (
    SculptokAPIError,
    SculptokClient,
    SculptokDepthMapParams,
    SculptokInsufficientCreditsError,
)
from mopa.settings import load_settings, resolve_sculptok_api_key

from .. import blob_store
from ..auth import get_optional_user, require_auth
from ..credits import InsufficientCreditsError, deduct_credit, get_credits
from ..queue import enqueue_sculptok, get_job_result
from ..schemas import (
    SculptokCreditsResponse,
    SculptokGenerateRequest,
    SculptokGenerateResponse,
)
from ..service_adapter import get_service, get_upload, heightmap_settings_to_dict

router = APIRouter(prefix="/sculptok", tags=["sculptok"])

_CACHE_DIR = Path(tempfile.gettempdir()) / "mopa_sculptok_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Job status schema
# --------------------------------------------------------------------------- #

class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "done", "failed"]
    result: Optional[SculptokGenerateResponse] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@router.get("/credits", response_model=SculptokCreditsResponse)
async def credits_route() -> SculptokCreditsResponse:
    client = _resolve_client()
    if client is None:
        return SculptokCreditsResponse(configured=False, balance=None)
    try:
        balance = client.get_credits()
    except SculptokAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Sculptok API error: {exc}") from exc
    return SculptokCreditsResponse(configured=True, balance=int(balance))


@router.post("/generate")
async def generate(
    req: SculptokGenerateRequest,
    user: Optional[dict] = Depends(get_optional_user),
) -> Any:
    """Generate a heightmap.

    Returns either:
      - {job_id} when Redis is available (async path)
      - SculptokGenerateResponse when running synchronously (fallback)
    """
    user_id: Optional[str] = user["sub"] if user else None

    # Credit gate: authenticated users must have credits remaining.
    if user_id:
        try:
            balance = get_credits(user_id)
        except Exception:
            balance = 999  # Supabase unavailable — don't block

        if balance <= 0:
            raise HTTPException(
                status_code=402,
                detail="No credits remaining. Visit /pricing to top up.",
            )

    # Prefer async via ARQ when Redis is configured.
    settings_dict = (
        heightmap_settings_to_dict(req.settings) if req.settings else None
    )
    job_id = await enqueue_sculptok(
        image_id=req.image_id,
        style=req.style,
        version=req.version,
        draw_hd=req.draw_hd,
        settings=settings_dict,
        user_id=user_id,
    )
    if job_id:
        return {"job_id": job_id}

    # ---------- synchronous fallback (no Redis) ----------
    return await _generate_sync(req, user_id=user_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def poll_job(job_id: str) -> JobStatusResponse:
    result = await get_job_result(job_id)
    if result is None:
        return JobStatusResponse(job_id=job_id, status="pending")
    if "error" in result:
        return JobStatusResponse(job_id=job_id, status="failed", error=result["error"])
    return JobStatusResponse(
        job_id=job_id,
        status="done",
        result=SculptokGenerateResponse(**result),
    )


# --------------------------------------------------------------------------- #
# Synchronous generation (local dev / Redis-absent fallback)
# --------------------------------------------------------------------------- #

async def _generate_sync(
    req: SculptokGenerateRequest,
    user_id: Optional[str],
) -> SculptokGenerateResponse:
    image = get_upload(req.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {req.image_id}")

    client = _resolve_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sculptok API key not configured. Set SCULPTOK_API_KEY env or "
                "add credentials.sculptok_api_key to ~/.mopa-heightmap/settings.json."
            ),
        )

    params = SculptokDepthMapParams(
        style=req.style, version=req.version, draw_hd=req.draw_hd,
    )

    sculptok_input: Image.Image = image
    subject_mask_id: Optional[str] = None
    sculptok_input_id: Optional[str] = None

    if req.settings is not None:
        svc = get_service()
        settings_dict = heightmap_settings_to_dict(req.settings)
        prepped, subject_alpha, _hash = svc.prepare_input(image, settings_dict)
        sculptok_input = prepped
        prepped_buf = io.BytesIO()
        prepped.save(prepped_buf, format="PNG")
        sculptok_input_id = blob_store.store_bytes(prepped_buf.getvalue(), "image/png")
        if subject_alpha is not None:
            arr16 = (np.clip(subject_alpha, 0.0, 1.0) * 65535).astype(np.uint16)
            mask_img = Image.fromarray(arr16, mode="I;16")
            mbuf = io.BytesIO()
            mask_img.save(mbuf, format="PNG")
            subject_mask_id = blob_store.store_bytes(mbuf.getvalue(), "image/png")

    photo_path = _CACHE_DIR / f"{req.image_id}_photo.png"
    sculptok_input.save(photo_path, format="PNG")
    out_path = _CACHE_DIR / f"{req.image_id}_sculptok.png"

    try:
        balance_before = client.get_credits()
    except SculptokAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Sculptok API error: {exc}") from exc

    try:
        result_path = client.generate_heightmap(
            photo_path, params=params, out_path=out_path, check_credits=False,
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

    # Deduct user credit on successful generation (best-effort).
    if user_id:
        try:
            deduct_credit(user_id)
        except (InsufficientCreditsError, Exception):
            pass

    return SculptokGenerateResponse(
        heightmap_path=str(result_path.resolve()),
        credits_used=max(0, balance_before - balance_after),
        credits_remaining=int(balance_after),
        sculptok_input_id=sculptok_input_id,
        subject_mask_id=subject_mask_id,
    )


def _resolve_client() -> SculptokClient | None:
    api_key = resolve_sculptok_api_key(cli_value=None, settings=load_settings())
    if not api_key:
        return None
    return SculptokClient(api_key)
