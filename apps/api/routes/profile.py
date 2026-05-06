"""GET / POST / DELETE /profiles — profile CRUD."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from mopa.profiles import (
    get_user_profiles_dir,
    list_profiles,
    resolve_profile_path,
    scaffold_profile,
)

from ..schemas import ProfileDetail, ProfileSaveRequest, ProfileSummary
from ..service_adapter import get_profile_data, list_profile_names

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=List[ProfileSummary])
async def list_profile_names_route() -> List[ProfileSummary]:
    names = list_profile_names()
    result: List[ProfileSummary] = []
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


@router.post("", response_model=ProfileDetail, status_code=201)
async def save_profile(req: ProfileSaveRequest) -> ProfileDetail:
    """Persist the current settings as a user-scope profile.

    Always writes to ``~/.mopa-heightmap/profiles/<name>.yaml`` (the
    user-scope dir) so it never overwrites the shipped material cards.
    Set ``overwrite=true`` to replace an existing user profile.
    """
    try:
        path = scaffold_profile(
            req.name,
            req.settings.model_dump(),
            machine=req.machine,
            lightburn_mode=req.lightburn_mode,
            black_is_deep=bool(req.settings.black_is_deep),
            overwrite=req.overwrite,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Profile {req.name!r} already exists; pass overwrite=true to replace.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProfileDetail(name=req.name, data=get_profile_data(req.name))


from fastapi import Response


@router.delete("/{name}", status_code=204, response_class=Response)
async def delete_profile(name: str) -> Response:
    """Delete a user-scope profile.

    Refuses to delete shipped (built-in) profiles to keep the system
    cards safe.
    """
    user_dir = get_user_profiles_dir()
    try:
        path = resolve_profile_path(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name!r}") from exc
    if path.parent.resolve() != user_dir.resolve():
        raise HTTPException(
            status_code=403,
            detail=f"Refusing to delete shipped profile {name!r}.",
        )
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(status_code=204)
