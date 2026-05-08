"""ARQ background worker — runs Sculptok jobs without blocking HTTP workers.

Start the worker alongside the API server:

    arq apps.api.worker.WorkerSettings

Requires Redis. Set REDIS_URL in the environment (defaults to localhost:6379).

The sculptok_generate task mirrors the logic in routes/sculptok.py but runs
asynchronously so the HTTP request returns immediately with a job_id.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from mopa.sculptok_client import (
    SculptokAPIError,
    SculptokClient,
    SculptokDepthMapParams,
    SculptokInsufficientCreditsError,
)
from mopa.settings import load_settings, resolve_sculptok_api_key

from . import blob_store
from .service_adapter import get_service, get_upload, heightmap_settings_to_dict

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
_CACHE_DIR = Path(tempfile.gettempdir()) / "mopa_sculptok_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

async def sculptok_generate_task(
    ctx: dict,
    *,
    image_id: str,
    style: str = "realistic",
    version: str = "v2",
    draw_hd: bool = True,
    settings: Optional[dict] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """ARQ task: generate a Sculptok heightmap for image_id.

    Returns a dict matching SculptokGenerateResponse fields so the
    polling endpoint can forward it to the frontend unchanged.
    """
    image = get_upload(image_id)
    if image is None:
        return {"error": f"Unknown image_id: {image_id}"}

    api_key = resolve_sculptok_api_key(cli_value=None, settings=load_settings())
    if not api_key:
        return {"error": "Sculptok API key not configured on the server."}

    client = SculptokClient(api_key)
    params = SculptokDepthMapParams(style=style, version=version, draw_hd=draw_hd)

    sculptok_input: Image.Image = image
    subject_mask_id: Optional[str] = None
    sculptok_input_id: Optional[str] = None

    if settings:
        svc = get_service()
        prepped, subject_alpha, _hash = svc.prepare_input(image, settings)
        sculptok_input = prepped
        buf = io.BytesIO()
        prepped.save(buf, format="PNG")
        sculptok_input_id = blob_store.store_bytes(buf.getvalue(), "image/png")
        if subject_alpha is not None:
            arr16 = (np.clip(subject_alpha, 0.0, 1.0) * 65535).astype(np.uint16)
            mask_img = Image.fromarray(arr16, mode="I;16")
            mbuf = io.BytesIO()
            mask_img.save(mbuf, format="PNG")
            subject_mask_id = blob_store.store_bytes(mbuf.getvalue(), "image/png")

    photo_path = _CACHE_DIR / f"{image_id}_photo.png"
    sculptok_input.save(photo_path, format="PNG")
    out_path = _CACHE_DIR / f"{image_id}_sculptok.png"

    try:
        balance_before = client.get_credits()
    except SculptokAPIError as exc:
        return {"error": f"Sculptok API error (credits): {exc}"}

    try:
        result_path = client.generate_heightmap(
            photo_path, params=params, out_path=out_path, check_credits=False,
        )
    except SculptokInsufficientCreditsError as exc:
        return {"error": str(exc), "code": "insufficient_credits"}
    except SculptokAPIError as exc:
        return {"error": f"Sculptok API error: {exc}"}
    except TimeoutError as exc:
        return {"error": str(exc), "code": "timeout"}

    try:
        balance_after = client.get_credits()
    except SculptokAPIError:
        balance_after = max(0, balance_before - params.expected_cost())

    # Deduct per-user credit if we have a user_id (SaaS mode).
    if user_id:
        try:
            from .credits import deduct_credit, InsufficientCreditsError
            deduct_credit(user_id)
        except Exception:
            pass  # best-effort — Sculptok already consumed the credit

    return {
        "heightmap_path": str(result_path.resolve()),
        "credits_used": max(0, balance_before - balance_after),
        "credits_remaining": int(balance_after),
        "sculptok_input_id": sculptok_input_id,
        "subject_mask_id": subject_mask_id,
    }


# --------------------------------------------------------------------------- #
# ARQ worker settings
# --------------------------------------------------------------------------- #

class WorkerSettings:
    functions = [sculptok_generate_task]
    redis_settings = _REDIS_URL  # arq accepts a URL string directly
    max_jobs = 4
    job_timeout = 300  # 5-minute hard cap per job
    keep_result = 3600  # keep job results in Redis for 1 hour
