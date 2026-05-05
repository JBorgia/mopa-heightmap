"""Application-level user settings persisted to ~/.mopa-heightmap/settings.json.

Distinct from per-export material profiles. These are operator preferences that
control how the app itself behaves (output naming, preview cap, default model,
device, UI port).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict


SETTINGS_VERSION = 1
DEFAULT_DIR = Path(os.path.expanduser("~")) / ".mopa-heightmap"
DEFAULT_PATH = DEFAULT_DIR / "settings.json"


@dataclass
class OutputSettings:
    directory: str = "outputs"
    naming: str = "overwrite"          # "overwrite" | "timestamp" | "counter"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    keep_history: bool = False
    layered: bool = False              # v1: single-PNG flow; v1.5b flips this on


@dataclass
class PreviewSettings:
    resolution_cap: int = 1024         # 0 = no cap
    flip_aug: bool = False
    auto_rerun_on_slider_change: bool = False  # v2


@dataclass
class InferenceSettings:
    default_model: str = "ZoeD_NK"
    device: str = "auto"               # "auto" | "cuda" | "cuda:0" | "cpu"
    flip_aug: bool = True              # for export, not preview
    pad_input: bool = True
    precision: str = "auto"            # "auto" | "fp32" | "fp16" | "bf16"
    inference_resolution: int = 0      # 0 = full; otherwise cap longest side fed to ZoeDepth
    # When True, models flagged ``requires_opt_in=True`` (e.g. DAv2-Large
    # under CC-BY-NC-4.0) appear in the model dropdown and may be downloaded.
    # Default off so the app never silently fetches non-commercial weights.
    allow_nc_weights: bool = False


@dataclass
class UiSettings:
    open_browser_on_launch: bool = True
    server_port: int = 7860
    theme: str = "default"


@dataclass
class AppSettings:
    version: int = SETTINGS_VERSION
    output: OutputSettings = field(default_factory=OutputSettings)
    preview: PreviewSettings = field(default_factory=PreviewSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    ui: UiSettings = field(default_factory=UiSettings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(target: Any, payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    for f in fields(target):
        if f.name not in payload:
            continue
        value = payload[f.name]
        current = getattr(target, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, f.name, value)


def load_settings(path: Path | None = None) -> AppSettings:
    """Load app settings, creating the file with defaults if missing."""
    settings_path = Path(path) if path else DEFAULT_PATH
    settings = AppSettings()
    if not settings_path.exists():
        try:
            save_settings(settings, settings_path)
        except OSError:
            pass
        return settings

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings

    _merge(settings, payload)
    return settings


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    settings_path = Path(path) if path else DEFAULT_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, settings_path)
    return settings_path


def resolve_device(setting: str) -> str:
    """Map an inference.device setting to a concrete torch device string."""
    import torch

    if setting and setting != "auto":
        return setting
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_precision(setting: str | None, device: str) -> str:
    """Map a precision setting to a concrete dtype label.

    "auto" picks fp16 on CUDA, fp32 on CPU. fp16/bf16 on CPU silently
    fall back to fp32 because ZoeDepth has ops (e.g. Conv2d) that have
    no half-precision implementation on CPU.
    """
    pref = (setting or "auto").lower()
    is_cuda = device.startswith("cuda")
    if pref == "auto":
        return "fp16" if is_cuda else "fp32"
    if pref in ("fp16", "bf16") and not is_cuda:
        return "fp32"
    if pref not in ("fp32", "fp16", "bf16"):
        return "fp32"
    return pref


def cuda_device_summary() -> str | None:
    """Return a short human-readable CUDA device summary, or None on CPU-only."""
    import torch

    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    try:
        free, total = torch.cuda.mem_get_info()
        return f"{name} · {free / 1e9:.1f} / {total / 1e9:.1f} GB free"
    except (RuntimeError, AttributeError):
        return name
