"""Re-run a previous export from its sidecar `*_settings.json`.

Schema produced by `exporter.write_settings_json` (see also `service.export`).
This module is the inverse: take that JSON and rebuild the inputs needed to
call `HeightmapService.export` again.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .service import DEFAULT_SETTINGS, ExportRequest, InferenceConfig


@dataclass
class RerunPayload:
    input_path: Path | None
    settings: dict
    inference: InferenceConfig
    profile_name: str | None
    profile_data: dict


def load_sidecar(path: str | Path) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Sidecar JSON must be an object: {path}")
    return data


def payload_from_sidecar(path: str | Path) -> RerunPayload:
    data = load_sidecar(path)

    settings = dict(DEFAULT_SETTINGS)
    settings.update(data.get("settings") or {})

    inf = data.get("inference") or {}
    inference = InferenceConfig(
        model_name=str(data.get("model") or inf.get("model") or "ZoeD_NK"),
        device=inf.get("device"),
        pad_input=bool(inf.get("pad_input", True)),
        with_flip_aug=bool(inf.get("with_flip_aug", False)),
        tile_size=int(inf.get("tile_size", 0)),
        tile_overlap=int(inf.get("tile_overlap", 128)),
    )

    raw_input = data.get("input")
    return RerunPayload(
        input_path=Path(raw_input) if raw_input else None,
        settings=settings,
        inference=inference,
        profile_name=data.get("profile"),
        profile_data=data.get("profile_data") or {},
    )


def request_for_sidecar(
    payload: RerunPayload,
    output_dir: Path,
    base_stem: str,
    *,
    naming: str = "counter",
    write_preview: bool = True,
) -> ExportRequest:
    """Build an ExportRequest that defaults to the safer 'counter' naming
    so a re-run never silently overwrites the original.
    """
    return ExportRequest(
        output_dir=output_dir,
        base_stem=base_stem,
        write_preview=write_preview,
        write_calibration_ramp=False,
        naming=naming,
    )
