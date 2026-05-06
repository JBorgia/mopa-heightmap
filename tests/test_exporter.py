import json
from pathlib import Path

import numpy as np
from PIL import Image

from mopa.exporter import (
    hash_image,
    resolve_export_stem,
    save_lightburn_png,
    save_master16_png,
    write_settings_json,
)


def _heightmap():
    return np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)


def test_save_lightburn_png_writes_uint8(tmp_path):
    target = tmp_path / "x_lightburn.png"
    save_lightburn_png(_heightmap(), target)
    assert target.exists()
    with Image.open(target) as im:
        assert im.mode == "L"
        assert im.size == (64, 64)


def test_save_master16_png_writes_uint16(tmp_path):
    target = tmp_path / "x_master16.png"
    save_master16_png(_heightmap(), target)
    with Image.open(target) as im:
        assert im.mode == "I;16"


def test_atomic_write_leaves_no_tmp(tmp_path):
    target = tmp_path / "x_lightburn.png"
    save_lightburn_png(_heightmap(), target)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_resolve_stem_overwrite_returns_base(tmp_path):
    assert resolve_export_stem(tmp_path, "coin", naming="overwrite") == "coin"
    assert resolve_export_stem(tmp_path, "coin_lightburn", naming="overwrite") == "coin"


def test_resolve_stem_timestamp_appends(tmp_path):
    stem = resolve_export_stem(tmp_path, "coin", naming="timestamp", timestamp_format="%Y")
    assert stem.startswith("coin_")
    assert len(stem) > len("coin_")


def test_resolve_stem_counter_increments(tmp_path):
    save_lightburn_png(_heightmap(), tmp_path / "coin_lightburn.png")
    nxt = resolve_export_stem(tmp_path, "coin", naming="counter")
    assert nxt == "coin_v2"
    save_lightburn_png(_heightmap(), tmp_path / "coin_v2_lightburn.png")
    assert resolve_export_stem(tmp_path, "coin", naming="counter") == "coin_v3"


def test_keep_history_forces_counter(tmp_path):
    save_lightburn_png(_heightmap(), tmp_path / "coin_lightburn.png")
    nxt = resolve_export_stem(tmp_path, "coin", naming="overwrite", keep_history=True)
    assert nxt == "coin_v2"


def test_hash_image_is_stable():
    a = Image.new("RGB", (8, 8), color=(10, 20, 30))
    assert hash_image(a) == hash_image(a.copy())


def test_write_settings_json_roundtrip(tmp_path):
    target = tmp_path / "x_settings.json"
    write_settings_json(
        target,
        input_path=tmp_path / "src.png",
        image_hash="deadbeef",
        device="cpu",
        model="ZoeD_NK",
        profile_name="mopa_60w_brass",
        profile_data={"name": "brass", "__profile_path__": "/tmp/x"},
        settings={"gamma": 0.8},
        inference={"pad_input": True},
        exports={"lightburn_png": "x_lightburn.png"},
        elapsed_s=1.234,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model"] == "ZoeD_NK"
    assert payload["profile"] == "mopa_60w_brass"
    assert "__profile_path__" not in payload["profile_data"]
    assert payload["elapsed_s"] == 1.234
    assert payload["timestamp_utc"]
