"""GET /profiles, POST /profiles — profile CRUD."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..schemas import ProfileDetail, ProfileSummary
from ..service_adapter import get_profile_data, list_profile_names

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=List[ProfileSummary])
async def list_profile_names_route() -> List[ProfileSummary]:
    names = list_profile_names()
    result = []
    for name in names:
        try:
            data = get_profile_data(name)
            result.append(ProfileSummary(
                name=name,
                machine=data.get("machine"),
                lightburn_mode=data.get("lightburn_mode"),
            ))
        except Exception:
            result.append(ProfileSummary(name=name))
    return result


@router.get("/{name}", response_model=ProfileDetail)
async def get_profile(name: str) -> ProfileDetail:
    try:
        data = get_profile_data(name)
        return ProfileDetail(name=name, data=data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name!r}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
