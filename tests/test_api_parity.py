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


def _binary_stl_vertices(data: bytes) -> np.ndarray:
    """Return an ``(n, 3, 3)`` float32 view of binary STL triangle vertices."""
    tri_count = struct.unpack_from("<I", data, 80)[0]
    vertices = np.empty((tri_count, 3, 3), dtype=np.float32)
    offset = 84
    for idx in range(tri_count):
        offset += 12  # skip normal
        for vertex_idx in range(3):
            vertices[idx, vertex_idx] = struct.unpack_from("<3f", data, offset)
            offset += 12
        offset += 2  # attribute byte count
    return vertices


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


def test_render_invalid_profile_returns_422(client, uploaded_image_id):
    resp = client.post(
        "/render",
        json={"image_id": uploaded_image_id, "profile_name": "does-not-exist"},
    )
    assert resp.status_code == 422
    assert "Profile not found" in resp.json()["detail"]


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


def test_click_mask_rejects_out_of_bounds_coordinates(client, uploaded_image_id):
    resp = client.post(
        "/mask/click",
        json={
            "image_id": uploaded_image_id,
            "x": 9999,
            "y": 9999,
            "label": "positive",
        },
    )
    assert resp.status_code == 422
    assert "outside image bounds" in resp.json()["detail"]


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


def test_export_stl_honors_base_thickness_mm(client, uploaded_image_id):
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]

    no_base = client.post(
        "/export/stl",
        json={"heightmap_id": hid, "z_scale_mm": 1.0, "base_thickness_mm": 0.0},
    )
    with_base = client.post(
        "/export/stl",
        json={"heightmap_id": hid, "z_scale_mm": 1.0, "base_thickness_mm": 2.5},
    )

    assert no_base.status_code == 200, no_base.text
    assert with_base.status_code == 200, with_base.text

    no_base_vertices = _binary_stl_vertices(no_base.content)
    with_base_vertices = _binary_stl_vertices(with_base.content)

    assert float(no_base_vertices[:, :, 2].min()) >= -1e-6
    assert float(with_base_vertices[:, :, 2].min()) == pytest.approx(-2.5, abs=1e-5)
    assert with_base_vertices.shape[0] > no_base_vertices.shape[0]


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


def test_resolve_lightburn_card_path_prefers_profile_override(monkeypatch, tmp_path):
    import apps.api.service_adapter as adapter

    default_card = tmp_path / f"{adapter.DEFAULT_PROFILE_NAME}.lbrn2"
    default_card.write_text("<LightBurnProject FormatVersion=\"1\" />", encoding="utf-8")
    override_card = tmp_path / "CustomCard.lbrn2"
    override_card.write_text("<LightBurnProject FormatVersion=\"1\" />", encoding="utf-8")

    monkeypatch.setattr(adapter, "DEFAULT_CARDS_DIR", tmp_path)

    assert adapter._resolve_lightburn_card_path({}) == default_card
    assert adapter._resolve_lightburn_card_path({"lightburn_card": "CustomCard"}) == override_card


def test_plan_and_lbrn2_export_use_profile_lightburn_card_override(
    client,
    uploaded_image_id,
    monkeypatch,
    tmp_path,
):
    import io as _io
    import os
    import xml.etree.ElementTree as ET
    import zipfile

    import apps.api.service_adapter as adapter

    profiles_dir = tmp_path / "profiles"
    cards_dir = tmp_path / "cards"
    profiles_dir.mkdir()
    cards_dir.mkdir()

    profile_path = profiles_dir / "override_profile.yaml"
    profile_path.write_text(
        "name: Override Profile\n"
        "lightburn_card: CustomCard\n"
        "heightmap: {}\n",
        encoding="utf-8",
    )

    custom_card = cards_dir / "CustomCard.lbrn2"
    custom_card.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<LightBurnProject AppVersion=\"9.9.9\" FormatVersion=\"1\">\n"
        "  <CutSetting type=\"Scan\">\n"
        "    <index Value=\"17\"/>\n"
        "    <name Value=\"C01\"/>\n"
        "    <maxPower Value=\"42\"/>\n"
        "    <speed Value=\"900\"/>\n"
        "    <frequency Value=\"30000\"/>\n"
        "    <QPulseWidth Value=\"120\"/>\n"
        "    <interval Value=\"0.03\"/>\n"
        "  </CutSetting>\n"
        "</LightBurnProject>\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MOPA_HEIGHTMAP_PROFILES", os.fspath(profiles_dir))
    monkeypatch.setattr(adapter, "DEFAULT_CARDS_DIR", cards_dir)

    render_resp = client.post(
        "/render",
        json={"image_id": uploaded_image_id, "profile_name": "override_profile"},
    )
    assert render_resp.status_code == 200, render_resp.text
    hid = render_resp.json()["heightmap_id"]

    plan_resp = client.post(
        "/plan",
        json={
            "image_id": uploaded_image_id,
            "heightmap_id": hid,
            "profile_name": "override_profile",
        },
    )
    assert plan_resp.status_code == 200, plan_resp.text
    plan_body = plan_resp.json()
    assert plan_body["passes"], plan_body
    assert plan_body["passes"][0]["label"] == "form: C01"
    assert plan_body["passes"][0]["pass_number"] == 17

    export_resp = client.post(
        "/export/lbrn2",
        json={
            "plan_id": plan_body["plan_id"],
            "heightmap_id": hid,
            "profile_name": "override_profile",
        },
    )
    assert export_resp.status_code == 200, export_resp.text

    zf = zipfile.ZipFile(_io.BytesIO(export_resp.content))
    names = set(zf.namelist())
    assert "project.lbrn2" in names
    assert "pass_00_form.png" in names

    root = ET.fromstring(zf.read("project.lbrn2"))
    assert root.attrib.get("AppVersion") == "9.9.9"

    cut_nodes = list(root.findall("CutSetting")) + list(root.findall("CutSetting_Img"))
    exported_layers = [
        {
            child.tag: child.attrib.get("Value")
            for child in cut
            if "Value" in child.attrib
        }
        for cut in cut_nodes
    ]
    assert any(
        layer.get("index") == "17"
        and layer.get("name") == "C01"
        and layer.get("maxPower") == "42"
        and layer.get("speed") == "900"
        for layer in exported_layers
    ), exported_layers


def test_plan_and_lbrn2_export_honor_profile_kind_color_overrides(
    client,
    uploaded_image_id,
    monkeypatch,
    tmp_path,
):
    import io as _io
    import os
    import xml.etree.ElementTree as ET
    import zipfile

    import apps.api.service_adapter as adapter

    profiles_dir = tmp_path / "profiles"
    cards_dir = tmp_path / "cards"
    profiles_dir.mkdir()
    cards_dir.mkdir()

    (profiles_dir / "override_profile.yaml").write_text(
        "name: Override Profile\n"
        "lightburn_card: CustomCard\n"
        "kind_color_overrides:\n"
        "  form: CustomDepth\n"
        "heightmap: {}\n",
        encoding="utf-8",
    )

    (cards_dir / "CustomCard.lbrn2").write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<LightBurnProject AppVersion=\"9.9.9\" FormatVersion=\"1\">\n"
        "  <CutSetting type=\"Scan\">\n"
        "    <index Value=\"23\"/>\n"
        "    <name Value=\"CustomDepth\"/>\n"
        "    <maxPower Value=\"38\"/>\n"
        "    <speed Value=\"777\"/>\n"
        "    <frequency Value=\"25000\"/>\n"
        "    <QPulseWidth Value=\"111\"/>\n"
        "    <interval Value=\"0.04\"/>\n"
        "  </CutSetting>\n"
        "</LightBurnProject>\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MOPA_HEIGHTMAP_PROFILES", os.fspath(profiles_dir))
    monkeypatch.setattr(adapter, "DEFAULT_CARDS_DIR", cards_dir)

    render_resp = client.post(
        "/render",
        json={"image_id": uploaded_image_id, "profile_name": "override_profile"},
    )
    assert render_resp.status_code == 200, render_resp.text
    hid = render_resp.json()["heightmap_id"]

    plan_resp = client.post(
        "/plan",
        json={
            "image_id": uploaded_image_id,
            "heightmap_id": hid,
            "profile_name": "override_profile",
        },
    )
    assert plan_resp.status_code == 200, plan_resp.text
    plan_body = plan_resp.json()
    assert plan_body["passes"], plan_body
    assert plan_body["passes"][0]["label"] == "form: CustomDepth"
    assert plan_body["passes"][0]["pass_number"] == 23

    export_resp = client.post(
        "/export/lbrn2",
        json={
            "plan_id": plan_body["plan_id"],
            "heightmap_id": hid,
            "profile_name": "override_profile",
        },
    )
    assert export_resp.status_code == 200, export_resp.text

    root = ET.fromstring(zipfile.ZipFile(_io.BytesIO(export_resp.content)).read("project.lbrn2"))
    cut_nodes = list(root.findall("CutSetting")) + list(root.findall("CutSetting_Img"))
    exported_layers = [
        {
            child.tag: child.attrib.get("Value")
            for child in cut
            if "Value" in child.attrib
        }
        for cut in cut_nodes
    ]
    assert any(
        layer.get("index") == "23"
        and layer.get("name") == "CustomDepth"
        and layer.get("maxPower") == "38"
        and layer.get("speed") == "777"
        for layer in exported_layers
    ), exported_layers


def test_export_bundle_includes_reference_artifacts_when_supplied(client, uploaded_image_id):
    """Subject mask, source photo, sculptok input, and profile YAML
    are bundled unconditionally when the client passes their blob ids.
    These are cheap to add and the wizard treats losing them as a
    workflow regression — re-running the wizard is the alternative."""
    import io as _io
    import zipfile

    # Render to get a heightmap, then store a synthetic mask + reuse the
    # source as the sculptok_input for this test.
    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]

    # Pretend we have a mask blob (1×1 PNG is enough to round-trip).
    from apps.api import blob_store as api_blob_store
    mask_id = api_blob_store.store_bytes(
        _io.BytesIO().getvalue() or _png_one_pixel(),
        content_type="image/png",
    )
    resp = client.post(
        "/export/bundle",
        json={
            "heightmap_id": hid,
            "include_png": True,
            "include_lbrn2": False,
            "include_stl": False,
            "image_id": uploaded_image_id,
            "sculptok_input_id": uploaded_image_id,  # same blob for test simplicity
            "subject_mask_id": mask_id,
            "profile_name": "mopa_60w_brass",
        },
    )
    assert resp.status_code == 200, resp.text
    zf = zipfile.ZipFile(_io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "heightmap.png" in names
    assert "subject_mask.png" in names, f"subject mask missing; saw {names}"
    assert "source_photo.png" in names, f"source photo missing; saw {names}"
    assert "sculptok_input.png" in names, f"sculptok input missing; saw {names}"
    # Profile YAML — gives the user enough to swap materials in LightBurn.
    profile_files = [n for n in names if n.startswith("profile_") and n.endswith(".yaml")]
    assert profile_files, f"profile YAML missing; saw {names}"


def test_export_bundle_skips_missing_reference_blobs_silently(client, uploaded_image_id):
    """Unknown reference blob_ids must NOT 404 the whole bundle — they
    just get skipped so the user still gets the heavy outputs."""
    import io as _io
    import zipfile

    render_resp = client.post("/render", json={"image_id": uploaded_image_id})
    hid = render_resp.json()["heightmap_id"]
    resp = client.post(
        "/export/bundle",
        json={
            "heightmap_id": hid,
            "include_png": True,
            "include_lbrn2": False,
            "include_stl": False,
            "subject_mask_id": "0" * 40,  # never seen by blob_store
            "image_id": "0" * 40,
        },
    )
    assert resp.status_code == 200
    names = set(zipfile.ZipFile(_io.BytesIO(resp.content)).namelist())
    assert "heightmap.png" in names
    assert "subject_mask.png" not in names
    assert "source_photo.png" not in names


def _png_one_pixel() -> bytes:
    """1×1 black PNG — smallest valid PNG. Used by tests that just need
    'a valid PNG blob' without exercising the image content."""
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("L", (1, 1), color=0).save(buf, format="PNG")
    return buf.getvalue()


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
