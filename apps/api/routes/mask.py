"""POST /mask, POST /mask/click — subject mask computation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import ClickMaskRequest, MaskRequest, MaskResponse
from ..service_adapter import do_click_mask, do_mask

router = APIRouter(prefix="/mask", tags=["mask"])


@router.post("", response_model=MaskResponse)
async def compute_mask(req: MaskRequest) -> MaskResponse:
    try:
        return do_mask(req.image_id, req.backend, req.edge_softness)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Mask backend '{req.backend}' requires missing package: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/click", response_model=MaskResponse)
async def click_mask(req: ClickMaskRequest) -> MaskResponse:
    try:
        return do_click_mask(
            image_id=req.image_id,
            mask_id=req.mask_id,
            x=req.x,
            y=req.y,
            label=req.label,
            clicker_key=req.clicker_key,
            tolerance=req.tolerance,
            max_fraction=req.max_fraction,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
