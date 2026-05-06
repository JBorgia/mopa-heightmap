"""POST /render — run depth inference + heightmap shaping."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import RenderRequest, RenderResponse
from ..service_adapter import do_render

router = APIRouter(prefix="/render", tags=["render"])


@router.post("", response_model=RenderResponse)
async def render(req: RenderRequest) -> RenderResponse:
    try:
        return do_render(
            image_id=req.image_id,
            settings=req.settings,
            profile_name=req.profile_name,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        # User-input problems (no heightmap path / file missing / invalid
        # settings) — 422 is the right HTTP code, not 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
