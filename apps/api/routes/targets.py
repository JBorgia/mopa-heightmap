"""GET /targets — list shipped target-object presets for the UI dropdown."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter

from mopa.target_presets import list_target_presets

from ..schemas import TargetPresetSummary


router = APIRouter(tags=["targets"])


@router.get("/targets", response_model=List[TargetPresetSummary])
async def targets() -> List[TargetPresetSummary]:
    return [
        TargetPresetSummary(
            name=p.name,
            display_name=p.display_name,
            print_width_mm=p.print_width_mm,
            print_height_mm=p.print_height_mm,
            polarity_invert=p.polarity_invert,
            notes=p.notes,
        )
        for p in list_target_presets()
    ]
