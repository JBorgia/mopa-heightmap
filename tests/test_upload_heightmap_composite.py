"""Tests for the Sculptok composite auto-crop in ``POST /upload/heightmap``.

Sculptok's web UI exports a 2:1.5-ish PNG with the depth map on the left
(black background) and a render preview on the right (grey background).
Engraving the whole thing carves the right-half preview into the
workpiece — so the upload route detects this layout and crops to the
depth-map half before persisting. These tests pin that behaviour and
the negative cases (single-image heightmap, square output).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from apps.api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_composite(half_w: int = 200, h: int = 280) -> bytes:
    """Synthetic side-by-side composite: same silhouette in each half on
    a black background, slightly different shading. The real Sculptok
    asset has a depth map next to a smaller preview rendering — we exercise
    that case in ``test_real_sculptok_composite_from_assets_dir_is_detected``;
    here we just pin the basic same-shape-twice detection path."""
    w = half_w * 2
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Left half: bright white subject on black.
    arr[h // 4 : 3 * h // 4, half_w // 4 : 3 * half_w // 4] = 220
    # Right half: same subject silhouette, slightly different brightness
    # (mimics depth-map vs render shading without flooding the threshold).
    arr[h // 4 : 3 * h // 4, half_w + half_w // 4 : half_w + 3 * half_w // 4] = 180
    return _png_bytes(Image.fromarray(arr, "RGB"))


def _make_single_landscape(w: int = 400, h: int = 200) -> bytes:
    """A real wide heightmap — both halves have the same dark background."""
    arr = np.zeros((h, w), dtype=np.uint8)
    # Single subject across the middle, no composite seam.
    cx, cy = w // 2, h // 2
    yy, xx = np.ogrid[:h, :w]
    blob = ((xx - cx) ** 2 / (w * 0.4) ** 2 + (yy - cy) ** 2 / (h * 0.4) ** 2) < 1
    arr[blob] = 200
    return _png_bytes(Image.fromarray(arr, "L"))


def _make_square(w: int = 256) -> bytes:
    arr = np.zeros((w, w), dtype=np.uint8)
    arr[w // 4 : 3 * w // 4, w // 4 : 3 * w // 4] = 200
    return _png_bytes(Image.fromarray(arr, "L"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_composite_is_detected_and_cropped_to_left_half(client):
    """The exact failure mode the user reported: side-by-side preview gets
    auto-cropped to just the depth-map half (the left one)."""
    resp = client.post(
        "/upload/heightmap",
        files={"file": ("preview.png", _make_composite(half_w=200, h=280), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["auto_cropped"] is True
    # 200 wide × 280 tall — the original file was 400 × 280.
    assert body["width"] == 200
    assert body["height"] == 280


def test_composite_cropped_file_is_actually_half_width_on_disk(client):
    """The persisted PNG must reflect the crop — render reads from disk."""
    # 400 × 240 = 1.67:1 — comfortably past the 1.4 aspect threshold.
    resp = client.post(
        "/upload/heightmap",
        files={"file": ("preview.png", _make_composite(half_w=200, h=240), "image/png")},
    )
    body = resp.json()
    assert body["auto_cropped"] is True
    on_disk = Image.open(Path(body["heightmap_path"]))
    assert on_disk.size == (200, 240)


def test_single_landscape_heightmap_is_not_cropped(client):
    """A wide-but-uniform heightmap (both halves' corners look the same)
    must pass through untouched."""
    resp = client.post(
        "/upload/heightmap",
        files={"file": ("wide.png", _make_single_landscape(400, 200), "image/png")},
    )
    body = resp.json()
    assert body["auto_cropped"] is False
    assert body["width"] == 400


def test_square_heightmap_is_not_cropped(client):
    """The most common Sculptok output (square 16-bit) must never be
    touched by the composite detector."""
    resp = client.post(
        "/upload/heightmap",
        files={"file": ("square.png", _make_square(256), "image/png")},
    )
    body = resp.json()
    assert body["auto_cropped"] is False
    assert body["width"] == 256
    assert body["height"] == 256


def test_real_sculptok_composite_from_assets_dir_is_detected(client):
    """Pin against the actual file the user uploaded so a future tuning
    change can't quietly break this exact case again."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "360_F_320748738_zddHlcaqbxBxOjXYpYpnQ4XlRT3cRS3H_sculptok.png"
    )
    if not fixture.exists():
        pytest.skip("Sculptok composite asset not present in this checkout")
    resp = client.post(
        "/upload/heightmap",
        files={"file": (fixture.name, fixture.read_bytes(), "image/png")},
    )
    body = resp.json()
    assert body["auto_cropped"] is True
    # Original is 1920×1280 → cropped to 960×1280 (left half).
    assert body["width"] == 960
    assert body["height"] == 1280


def test_composite_crops_to_darker_half_when_layout_is_reversed(client):
    """Same subject, but the brighter (preview-shaded) half is on the
    LEFT and the dimmer (depth-map-shaded) half is on the RIGHT. The
    detector keeps the half with the darker mean — that's the depth map."""
    half_w, h = 200, 280
    w = half_w * 2
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # LEFT half: brighter (preview-like)
    arr[h // 4 : 3 * h // 4, half_w // 4 : 3 * half_w // 4] = 240
    # RIGHT half: dimmer (depth-map-like)
    arr[h // 4 : 3 * h // 4, half_w + half_w // 4 : half_w + 3 * half_w // 4] = 160
    resp = client.post(
        "/upload/heightmap",
        files={"file": ("rev.png", _png_bytes(Image.fromarray(arr, "RGB")), "image/png")},
    )
    body = resp.json()
    assert body["auto_cropped"] is True
    on_disk = Image.open(Path(body["heightmap_path"]))
    assert on_disk.size == (200, 280)
    # The kept half is the darker (right) one.
    arr_on_disk = np.asarray(on_disk.convert("L"))
    assert arr_on_disk.max() <= 160
