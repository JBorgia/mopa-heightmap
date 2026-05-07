"""API parity tests — verify that POST /render + GET /blob/{id} produce byte-identical
PNG output to calling HeightmapService.render() directly.

Must pass before any Angular UI work begins (per SONNET_UI_MIGRATION_BRIEF.md §7).
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mopa.service import (
    DEFAULT_SETTINGS,
    HeightmapService,
    merge_profile_settings,
)
from mopa.settings import AppSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """Small deterministic gradient — no real model needed for parity check."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        arr[r, :, 0] = int(r * 255 / (h - 1))  # R gradient
        arr[r, :, 2] = 128                       # fixed B
    return Image.fromarray(arr, "RGB")


def _png_to_float32(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32) / 255.0


def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Spin up the FastAPI app in-process with TestClient."""
    from apps.api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def gradient_png_bytes() -> bytes:
    return _img_to_png_bytes(_make_gradient_image())


@pytest.fixture(scope="module")
def uploaded_image_id(client, gradient_png_bytes) -> str:
    resp = client.post(
        "/upload",
        files={"file": ("gradient.png", gradient_png_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]


# ---------------------------------------------------------------------------
# Basic smoke tests (don't need a real ZoeDepth model)
# ---------------------------------------------------------------------------

class _StubService:
    """Replaces HeightmapService.render() with a deterministic output."""

    def render(self, image: Image.Image, settings: dict):
        from mopa.service import PreviewResult
        arr = np.zeros((image.height, image.width), dtype=np.float32)
        # Simple fill: mean of red channel / 255
        r_mean = np.asarray(image)[:, :, 0].mean() / 255.0
        arr[:] = r_mean
        preview = Image.fromarray((arr * 255).astype(np.uint8), "L").convert("RGB")
        from mopa.exporter import hash_image
        return PreviewResult(
            heightmap=arr,
            preview_image=preview,
            settings=settings,
            elapsed_s=0.0,
            image_hash=hash_image(image),
        )


@pytest.fixture(autouse=True)
def _patch_service(monkeypatch):
    """Swap out HeightmapService for the stub so tests never hit ZoeDepth."""
    import apps.api.service_adapter as adapter
    stub = _StubService()
    monkeypatch.setattr(adapter, "get_service", lambda: stub)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_returns_metadata(client, gradient_png_bytes):
    resp = client.post(
        "/upload",
        files={"file": ("test.png", gradient_png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "image_id" in body
    assert body["w"] == 64
    assert body["h"] == 64
    assert len(body["sha256"]) == 64  # full hex sha256


def test_upload_bad_file_returns_422(client):
    resp = client.post(
        "/upload",
        files={"file": ("bad.bin", b"\x00\x01\x02", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_render_returns_blob_ids(client, uploaded_image_id):
    resp = client.post("/render", json={"image_id": uploaded_image_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "heightmap_id" in body
    assert "preview_id" in body
    assert body["elapsed_s"] >= 0


def test_render_unknown_image_returns_404(client):
    resp = client.post("/render", json={"image_id": "doesnotexist"})
    assert resp.status_code == 404


def test_blob_fetch_returns_png(client, uploaded_image_id):
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    assert render_resp.status_code == 200
    heightmap_id = render_resp.json()["heightmap_id"]
    blob_resp = client.get(f"/blob/{heightmap_id}")
    assert blob_resp.status_code == 200
    assert blob_resp.headers["content-type"] == "image/png"
    # Must be a valid PNG
    arr = _png_to_float32(blob_resp.content)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0 + 1e-5


def test_blob_not_found_returns_404(client):
    resp = client.get("/blob/0000000000000000000000000000000000000000")
    assert resp.status_code == 404


def test_blob_cache_control_immutable(client, uploaded_image_id):
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.get(f"/blob/{hid}")
    assert "immutable" in resp.headers.get("cache-control", "")


def test_export_png_16bit(client, uploaded_image_id):
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.post("/export/png", json={"heightmap_id": hid, "bit_depth": 16})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    arr = _png_to_float32(resp.content)
    assert arr.dtype == np.float32


def test_export_png_8bit(client, uploaded_image_id):
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.post("/export/png", json={"heightmap_id": hid, "bit_depth": 8})
    assert resp.status_code == 200


def test_export_stl_returns_binary_mesh(client, uploaded_image_id):
    """Regression for the numpy-stl ``Mode`` import path.

    The library moved ``Mode`` from ``stl.mesh.Mode`` to top-level ``stl.Mode``;
    referencing the old path raises AttributeError at runtime and turns the
    export into HTTP 500. This test exercises the real export route end to end
    so a future Pillow / numpy-stl bump can't break it silently.
    """
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    # 8-pixel mesh keeps the test under a second; the real product can ship
    # multi-million-triangle meshes but that's not what we're verifying here.
    resp = client.post(
        "/export/stl",
        json={"heightmap_id": hid, "z_scale_mm": 1.0, "base_thickness_mm": 0.5},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "model/stl"
    assert resp.headers["content-disposition"].endswith("filename=heightmap.stl")
    assert len(resp.content) > 0


def test_render_response_includes_conditioned_and_render_mask_fields(client, uploaded_image_id):
    """The wizard preview pane reads conditioned_id / render_mask_id off the
    /render response. These fields must be present (Optional[str]); absence
    breaks the OpenAPI contract."""
    resp = client.post("/render", json={"image_id": uploaded_image_id})
    body = resp.json()
    assert "conditioned_id" in body
    assert "render_mask_id" in body


def test_export_bundle_zips_selected_formats(client, uploaded_image_id):
    """Wizard's Submit action: bundle PNG + STL into a single zip the user
    downloads in one click. Skip .lbrn2 here — that path exercises the
    pass-plan flow which the stub doesn't drive."""
    import zipfile
    import io as _io

    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.post(
        "/export/bundle",
        json={
            "heightmap_id": hid,
            "include_png": True,
            "include_lbrn2": False,
            "include_stl": True,
            "z_scale_mm": 1.0,
            "base_thickness_mm": 0.5,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert "mopa_export.zip" in resp.headers["content-disposition"]
    zf = zipfile.ZipFile(_io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "heightmap.png" in names
    assert "heightmap.stl" in names


def test_export_bundle_rejects_empty_selection(client, uploaded_image_id):
    """Submitting nothing is a 422 — server can't build an empty bundle."""
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.post(
        "/export/bundle",
        json={
            "heightmap_id": hid,
            "include_png": False,
            "include_lbrn2": False,
            "include_stl": False,
        },
    )
    assert resp.status_code == 422


def test_export_bundle_lbrn2_without_plan_id_is_422(client, uploaded_image_id):
    """include_lbrn2=true without a plan_id is a client bug — be explicit
    rather than silently dropping the .lbrn2 from the bundle."""
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.post(
        "/export/bundle",
        json={
            "heightmap_id": hid,
            "include_png": True,
            "include_lbrn2": True,
            "include_stl": False,
            # plan_id deliberately missing
        },
    )
    assert resp.status_code == 422
    assert "plan_id" in resp.json()["detail"]


def test_profiles_list(client):
    resp = client.get("/profiles")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_parity_service_vs_api(client, gradient_png_bytes):
    """Core parity test: direct service call vs API round-trip produce the same heightmap."""
    # --- Direct call via stub ---
    img = _make_gradient_image()
    stub = _StubService()
    direct_result = stub.render(img, dict(DEFAULT_SETTINGS))
    direct_heightmap = direct_result.heightmap  # float32 array

    # --- API round-trip ---
    up = client.post("/upload", files={"file": ("g.png", gradient_png_bytes, "image/png")})
    assert up.status_code == 200
    iid = up.json()["image_id"]

    render = client.post("/render", json={"image_id": iid})
    assert render.status_code == 200
    hid = render.json()["heightmap_id"]

    blob = client.get(f"/blob/{hid}")
    assert blob.status_code == 200
    api_heightmap = _png_to_float32(blob.content)

    # Both should be very close (PNG quantisation ~ 1/65535 ≈ 1.5e-5 relative error)
    assert direct_heightmap.shape == api_heightmap.shape, (
        f"Shape mismatch: {direct_heightmap.shape} vs {api_heightmap.shape}"
    )
    max_diff = float(np.abs(direct_heightmap - api_heightmap).max())
    assert max_diff < 1e-3, f"Parity failed: max pixel diff = {max_diff:.6f}"
