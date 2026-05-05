"""POST /plan — wrap plan_passes() for the Angular client."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import service_adapter
from ..schemas import PassPlanRequest, PassPlanResponse

router = APIRouter(tags=["plan"])


@router.post("/plan", response_model=PassPlanResponse)
async def create_plan(req: PassPlanRequest) -> PassPlanResponse:
    """Compute an ordered pass stack from a rendered heightmap + material profile."""
    try:
        return service_adapter.do_plan(
            image_id=req.image_id,
            heightmap_id=req.heightmap_id,
            profile_name=req.profile_name,
            settings=req.settings,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
