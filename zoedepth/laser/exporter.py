"""Export bundle writer: atomic writes, naming modes, sidecar JSON."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image

from .heightmap import to_uint8, to_uint16

APP_VERSION = "0.2.0"


@dataclass
class ExportPaths:
    lightburn_png: Path
    master16_png: Path
    preview_png: Path | None
    ramp_png: Path | None
    settings_json: Path


@dataclass
class ExportBundle:
    paths: ExportPaths
    elapsed_s: float
    stem: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _strip_known_suffix(stem: str) -> str:
    for suffix in ("_lightburn", "_master16", "_preview", "_ramp", "_settings"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def _atomic_save_image(image: Image.Image, target: Path) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    # PIL can't infer format from a .tmp suffix — derive it from the real target.
    fmt = (target.suffix or ".png").lstrip(".").upper()
    if fmt in {"JPG"}:
        fmt = "JPEG"
    image.save(tmp, format=fmt)
    os.replace(tmp, target)


def _next_counter_stem(directory: Path, base_stem: str) -> str:
    """Find the smallest _vN suffix not yet present for any export file."""
    pattern = re.compile(rf"^{re.escape(base_stem)}(?:_v(\d+))?_(?:lightburn|master16|preview|ramp|settings)\.")
    highest = 1
    if directory.exists():
        for child in directory.iterdir():
            match = pattern.match(child.name)
            if not match:
                continue
            n = int(match.group(1)) if match.group(1) else 1
            if n >= highest:
                highest = n + 1
    if highest == 1:
        # If no prior exports exist, use plain base_stem.
        return base_stem
    return f"{base_stem}_v{highest}"


def resolve_export_stem(
    directory: Path,
    base_stem: str,
    naming: str = "overwrite",
    timestamp_format: str = "%Y%m%d_%H%M%S",
    keep_history: bool = False,
) -> str:
    """Derive a final stem based on naming policy and existing files."""
    base = _strip_known_suffix(base_stem)
    if keep_history or naming == "counter":
        return _next_counter_stem(directory, base)
    if naming == "timestamp":
        ts = _dt.datetime.now(_dt.timezone.utc).strftime(timestamp_format)
        return f"{base}_{ts}"
    return base


def hash_image(image: Image.Image) -> str:
    """Stable short hash of an image's RGB bytes."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    return digest[:16]


def save_lightburn_png(heightmap: np.ndarray, path: Path) -> Path:
    img = Image.fromarray(to_uint8(heightmap), mode="L")
    _atomic_save_image(img, path)
    return path


def save_master16_png(heightmap: np.ndarray, path: Path) -> Path:
    img = Image.fromarray(to_uint16(heightmap), mode="I;16")
    _atomic_save_image(img, path)
    return path


def save_preview_png(image: Image.Image, path: Path) -> Path:
    _atomic_save_image(image, path)
    return path


def save_ramp_png(image: Image.Image, path: Path) -> Path:
    _atomic_save_image(image, path)
    return path


def write_settings_json(
    path: Path,
    *,
    input_path: str | os.PathLike | None,
    image_hash: str,
    device: str,
    model: str,
    profile_name: str | None,
    profile_data: Mapping[str, Any],
    settings: Mapping[str, Any],
    inference: Mapping[str, Any],
    exports: Mapping[str, Any],
    elapsed_s: float,
) -> Path:
    payload = {
        "app_version": APP_VERSION,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "input": str(input_path) if input_path else None,
        "input_sha256_16": image_hash,
        "device": device,
        "model": model,
        "profile": profile_name,
        "profile_data": {k: v for k, v in profile_data.items() if k != "__profile_path__"},
        "settings": dict(settings),
        "inference": dict(inference),
        "exports": dict(exports),
        "elapsed_s": round(float(elapsed_s), 4),
    }
    _atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))
    return path
