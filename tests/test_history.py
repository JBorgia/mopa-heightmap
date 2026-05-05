"""Tests for the JSONL recipe history."""
from __future__ import annotations

import json
from pathlib import Path

from zoedepth.laser.history import (
    HISTORY_PATH_ENV,
    HistoryEntry,
    append_history,
    make_entry,
    read_history,
)


def _entry(stem="x") -> HistoryEntry:
    return make_entry(
        image_hash="deadbeef",
        settings={"gamma": 0.7},
        inference={"model": "ZoeD_NK", "device": "cpu"},
        output_dir=Path("/tmp/out"),
        stem=stem,
        elapsed_s=1.25,
        input_path="/tmp/in.png",
        profile="anodized_aluminum",
        model="ZoeD_NK",
        device="cpu",
    )


def test_make_entry_round_trip(tmp_path: Path):
    target = tmp_path / "history.jsonl"
    append_history(_entry("alpha"), path=target)
    append_history(_entry("beta"), path=target)

    rows = read_history(path=target)
    assert len(rows) == 2
    assert rows[0]["stem"] == "alpha"
    assert rows[1]["stem"] == "beta"
    assert rows[0]["image_hash"] == "deadbeef"
    assert rows[0]["profile"] == "anodized_aluminum"
    assert rows[0]["settings"] == {"gamma": 0.7}


def test_read_history_missing_file_returns_empty(tmp_path: Path):
    assert read_history(path=tmp_path / "nope.jsonl") == []


def test_read_history_limit(tmp_path: Path):
    target = tmp_path / "h.jsonl"
    for i in range(5):
        append_history(_entry(f"s{i}"), path=target)
    rows = read_history(path=target, limit=2)
    assert [r["stem"] for r in rows] == ["s3", "s4"]


def test_env_override_redirects(tmp_path: Path, monkeypatch):
    target = tmp_path / "env.jsonl"
    monkeypatch.setenv(HISTORY_PATH_ENV, str(target))
    append_history(_entry("env"))
    assert target.exists()
    rows = read_history()
    assert rows[0]["stem"] == "env"


def test_history_skips_corrupt_lines(tmp_path: Path):
    target = tmp_path / "mixed.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"stem": "ok", "version": 1}) + "\nnot json\n",
        encoding="utf-8",
    )
    rows = read_history(path=target)
    assert len(rows) == 1
    assert rows[0]["stem"] == "ok"
