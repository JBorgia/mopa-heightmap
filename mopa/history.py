"""Append-only JSONL recipe history.

Every successful export drops one line into ~/.mopa-heightmap/history.jsonl
(or a configurable path) so the operator can audit what they ran, when,
against which input, with which profile.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_HISTORY_PATH = Path(os.path.expanduser("~")) / ".mopa-heightmap" / "history.jsonl"
HISTORY_VERSION = 1
HISTORY_PATH_ENV = "MOPA_HEIGHTMAP_HISTORY"


def _resolved_default() -> Path:
    override = os.environ.get(HISTORY_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_HISTORY_PATH


@dataclass
class HistoryEntry:
    timestamp_utc: str
    input: str | None
    image_hash: str
    profile: str | None
    model: str
    device: str
    output_dir: str
    stem: str
    elapsed_s: float
    settings: dict
    inference: dict

    def to_dict(self) -> dict:
        d = {
            "version": HISTORY_VERSION,
            "timestamp_utc": self.timestamp_utc,
            "input": self.input,
            "image_hash": self.image_hash,
            "profile": self.profile,
            "model": self.model,
            "device": self.device,
            "output_dir": self.output_dir,
            "stem": self.stem,
            "elapsed_s": round(float(self.elapsed_s), 4),
            "settings": dict(self.settings),
            "inference": dict(self.inference),
        }
        return d


def append_history(entry: HistoryEntry, path: Path | None = None) -> Path:
    target = Path(path) if path else _resolved_default()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
    # Append-only is implicitly atomic for one-line writes on Windows + POSIX.
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return target


def read_history(path: Path | None = None, *, limit: int | None = None) -> list[dict]:
    target = Path(path) if path else _resolved_default()
    if not target.exists():
        return []
    rows: list[dict] = []
    with target.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        rows = rows[-int(limit):]
    return rows


def make_entry(
    *,
    image_hash: str,
    settings: Mapping[str, Any],
    inference: Mapping[str, Any],
    output_dir: Path,
    stem: str,
    elapsed_s: float,
    input_path: str | os.PathLike | None,
    profile: str | None,
    model: str,
    device: str,
) -> HistoryEntry:
    return HistoryEntry(
        timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        input=str(input_path) if input_path else None,
        image_hash=image_hash,
        profile=profile,
        model=model,
        device=device,
        output_dir=str(output_dir),
        stem=stem,
        elapsed_s=elapsed_s,
        settings=dict(settings),
        inference=dict(inference),
    )
