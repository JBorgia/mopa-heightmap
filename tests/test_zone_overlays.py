"""Tests for zone overlay generators (rim/border/field decorations).

Covers every rim + border pattern, the field pattern's subject-mask
protection, and the /render mask_id reuse path end-to-end.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from mopa.backgrounds import basket_weave_pattern
from mopa.zone_overlays import (
    acanthus_wave_overlay,
    apply_zone_overlays,
    beaded_rim_overlay,
    denticled_rim_overlay,
    greek_key_overlay,
    laurel_wreath_overlay,
    reeded_rim_overlay,
    rope_rim_overlay,
    rope_twist_overlay,
    serrated_rim_overlay,
)

H = W = 240
GEO = dict(cx=W / 2, cy=H / 2, outer_r_px=110.0, rim_width_px=10.0)
BORDER = dict(**GEO, border_width_px=22.0)


# ----------------------------------------------------------- raw overlays

@pytest.mark.parametrize("fn,kwargs", [
    (beaded_rim_overlay, GEO),
    (reeded_rim_overlay, GEO),
    (denticled_rim_overlay, GEO),
    (rope_rim_overlay, GEO),
    (serrated_rim_overlay, GEO),
    (rope_twist_overlay, BORDER),
    (acanthus_wave_overlay, BORDER),
    (greek_key_overlay, BORDER),
    (laurel_wreath_overlay, BORDER),
])
def test_overlay_shape_range_and_nonflat(fn, kwargs):
    arr = fn(H, W, **kwargs)
    assert arr.shape == (H, W)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    # Every decorative overlay must actually contain relief.
    assert arr.max() - arr.min() > 0.5


def test_greek_key_repeats_around_annulus():
    a = greek_key_overlay(H, W, **BORDER, key_count=8)
    b = greek_key_overlay(H, W, **BORDER, key_count=32)
    # More repeats = different pattern.
    assert not np.allclose(a, b)


def test_reeded_rim_reed_count_changes_pattern():
    a = reeded_rim_overlay(H, W, **GEO, reed_count=40)
    b = reeded_rim_overlay(H, W, **GEO, reed_count=160)
    assert not np.allclose(a, b)


def test_basket_weave_has_alternating_cells():
    arr = basket_weave_pattern(128, 128, scale=1.0)
    assert arr.shape == (128, 128)
    assert 0.0 <= arr.min() and arr.max() <= 1.0
    # Weave must not be a flat fill or simple stripes: both a horizontal
    # and a vertical slice through cell centres should oscillate.
    assert arr[32, :].std() > 0.05
    assert arr[:, 32].std() > 0.05


# ----------------------------------------------------------- dispatch

def _flat_hm() -> np.ndarray:
    return np.full((H, W), 0.5, dtype=np.float32)


@pytest.mark.parametrize("settings", [
    {"rim_pattern": "beaded"},
    {"rim_pattern": "reeded"},
    {"rim_pattern": "denticled"},
    {"rim_pattern": "rope"},
    {"rim_pattern": "serrated"},
    {"border_pattern": "rope_twist"},
    {"border_pattern": "acanthus_wave"},
    {"border_pattern": "greek_key"},
    {"border_pattern": "laurel"},
    {"field_pattern": "basket_weave"},
    {"field_pattern": "guilloche"},
])
def test_apply_zone_overlays_changes_pixels(settings):
    base = {"zone_width_mm": 24.0, "zone_height_mm": 24.0, "zone_shape": "circle"}
    out = apply_zone_overlays(_flat_hm(), {**base, **settings})
    assert (out != 0.5).sum() > 100, f"{settings} produced no visible change"


def test_apply_zone_overlays_noop_without_geometry():
    out = apply_zone_overlays(_flat_hm(), {"border_pattern": "greek_key"})
    assert np.array_equal(out, _flat_hm())


def test_field_pattern_spares_subject_when_alpha_given():
    base = {
        "zone_width_mm": 24.0, "zone_height_mm": 24.0, "zone_shape": "circle",
        "field_pattern": "guilloche",
    }
    # Subject occupies the central disc.
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    subject = (np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) < 40).astype(np.float32)

    unprotected = apply_zone_overlays(_flat_hm(), dict(base))
    protected = apply_zone_overlays(_flat_hm(), dict(base), subject_alpha=subject)

    inside = subject >= 0.5
    # Without alpha the pattern stomps the subject area; with alpha it must not.
    assert (unprotected[inside] != 0.5).sum() > 100
    assert np.allclose(protected[inside], 0.5, atol=1e-4)
    # The field outside the subject is still patterned.
    assert (protected[~inside] != 0.5).sum() > 100


# ----------------------------------------------------------- export-time layers

def test_zone_pattern_layer_builds_relief_per_zone():
    from mopa.zone_overlays import zone_pattern_layer
    settings = {
        "zone_border_width_mm": 3.0, "zone_rim_width_mm": 1.5,
        "field_pattern": "basket_weave",
        "border_pattern": "greek_key",
        "rim_pattern": "beaded",
    }
    for zone in ("field", "border", "rim"):
        layer = zone_pattern_layer(zone, settings, H, W, print_w_mm=30.0, print_h_mm=30.0)
        assert layer is not None, zone
        assert layer.shape == (H, W)
        assert layer.max() - layer.min() > 0.3, f"{zone} layer is flat"


def test_zone_pattern_layer_none_when_unconfigured():
    from mopa.zone_overlays import zone_pattern_layer
    for zone in ("field", "border", "rim"):
        assert zone_pattern_layer(zone, {}, H, W, print_w_mm=30.0, print_h_mm=30.0) is None


def test_mask_from_heightmap_extracts_subject():
    from mopa.zones import mask_from_heightmap
    hm = np.zeros((200, 200), dtype=np.float32)  # flat black background
    hm[60:140, 60:140] = 0.8                     # raised subject block
    mask = mask_from_heightmap(hm)
    assert mask is not None
    assert mask[100, 100] > 0.9
    assert mask[10, 10] < 0.1
    # Coverage ≈ the block's area.
    assert 0.1 < float(mask.mean()) < 0.3


def test_mask_from_heightmap_rejects_unflat_background():
    from mopa.zones import mask_from_heightmap
    rng = np.random.default_rng(1)
    hm = rng.uniform(0, 1, (200, 200)).astype(np.float32)  # noise everywhere
    assert mask_from_heightmap(hm) is None


def test_decorated_zones_sculpt_at_scaled_pass_counts():
    """A zone with a pattern selected must engrave REAL relief: device
    parameters with numPasses scaled by the pattern depth slider — not the
    profile's 2-pass anneal (that's a tinted stencil, not 3D)."""
    import io as _io
    import re as _re
    import zipfile as _zipfile
    from fastapi.testclient import TestClient
    from apps.api.main import app

    client = TestClient(app)
    rng = np.random.default_rng(5)
    photo = rng.uniform(60, 200, (200, 200, 3)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(photo).save(buf, format="PNG")
    image_id = client.post("/upload", files={"file": ("p.png", buf.getvalue(), "image/png")}).json()["image_id"]

    hm = np.zeros((200, 200), dtype=np.float32)
    hm[60:140, 60:140] = 0.8
    buf = io.BytesIO(); Image.fromarray((hm * 65535).astype(np.uint16)).save(buf, format="PNG")
    hm_path = client.post("/upload/heightmap", files={"file": ("h.png", buf.getvalue(), "image/png")}).json()["heightmap_path"]

    settings = {
        "external_heightmap_path": hm_path, "heightmap_enhance_mode": "off",
        "zone_width_mm": 60.0, "zone_height_mm": 60.0, "zone_shape": "circle",
        "field_pattern": "guilloche", "field_pattern_depth": 0.5,
        "border_pattern": "greek_key", "border_pattern_depth": 0.75,
    }
    r = client.post("/render", json={"image_id": image_id, "settings": settings,
                                     "profile_name": "mopa_60w_brass"})
    assert r.status_code == 200, r.text
    plan = client.post("/plan", json={"image_id": image_id, "heightmap_id": r.json()["heightmap_id"],
                                      "profile_name": "mopa_60w_brass", "settings": settings,
                                      "shape_override": "circle"}).json()
    exp = client.post("/export/lbrn2", json={"plan_id": plan["plan_id"],
                                             "heightmap_id": r.json()["heightmap_id"],
                                             "profile_name": "mopa_60w_brass",
                                             "shape_override": "circle"})
    assert exp.status_code == 200, exp.text
    xml = _zipfile.ZipFile(_io.BytesIO(exp.content)).read("project.lbrn2").decode()

    def layer(name):
        m = _re.search(rf'<name Value="{name}"/>(.*?)</CutSetting_Img>', xml, _re.S)
        assert m, f"layer {name} missing"
        return m.group(1)

    # Brass sculpt = 2000 mm/s @ 512 passes. Field 0.5 → 256, border 0.75 → 384.
    fld = layer("ZoneField")
    assert '<numPasses Value="256"/>' in fld and '<speed Value="2000"/>' in fld
    brd = layer("ZoneBorder")
    assert '<numPasses Value="384"/>' in brd and '<speed Value="2000"/>' in brd
    # Rim has NO pattern selected → keeps the profile's anneal parameters.
    rim = layer("ZoneRim")
    assert '<numPasses Value="2"/>' in rim and '<speed Value="1800"/>' in rim


def test_zone_precedence_and_exergue_gating():
    """Zone precedence: rings and the exergue strip clip the device (the
    frame wins; the strip is a reserved text panel), the field yields to
    the device, and the exergue pass only exists when exergue_enabled —
    when disabled its strip belongs to the field."""
    import io as _io
    import zipfile as _zipfile
    from fastapi.testclient import TestClient
    from apps.api.main import app

    client = TestClient(app)
    rng = np.random.default_rng(9)
    photo = rng.uniform(60, 200, (240, 240, 3)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(photo).save(buf, format="PNG")
    image_id = client.post("/upload", files={"file": ("p.png", buf.getvalue(), "image/png")}).json()["image_id"]

    # Subject block reaches into the bottom exergue strip (rows > 197)
    # without touching the canvas edge (which would defeat the flat-border
    # background check).
    hm = np.zeros((240, 240), dtype=np.float32)
    hm[80:224, 80:160] = 0.8
    buf = io.BytesIO(); Image.fromarray((hm * 65535).astype(np.uint16)).save(buf, format="PNG")
    hm_path = client.post("/upload/heightmap", files={"file": ("h.png", buf.getvalue(), "image/png")}).json()["heightmap_path"]

    base = {"external_heightmap_path": hm_path, "heightmap_enhance_mode": "off",
            "zone_width_mm": 60.0, "zone_height_mm": 60.0, "zone_shape": "circle"}

    def run(settings):
        r = client.post("/render", json={"image_id": image_id, "settings": settings,
                                         "profile_name": "mopa_60w_brass"})
        assert r.status_code == 200, r.text
        plan = client.post("/plan", json={"image_id": image_id, "heightmap_id": r.json()["heightmap_id"],
                                          "profile_name": "mopa_60w_brass", "settings": settings,
                                          "shape_override": "circle"}).json()
        exp = client.post("/export/lbrn2", json={"plan_id": plan["plan_id"],
                                                 "heightmap_id": r.json()["heightmap_id"],
                                                 "profile_name": "mopa_60w_brass",
                                                 "shape_override": "circle",
                                                 "clean_heightmap_id": r.json().get("clean_heightmap_id")})
        assert exp.status_code == 200, exp.text
        return plan, _zipfile.ZipFile(_io.BytesIO(exp.content))

    # Default: exergue pass absent; its strip belongs to the field, so the
    # device still engraves the subject's bottom (inside the strip area).
    plan, zf = run(dict(base))
    assert not any("Exergue" in p["label"] for p in plan["passes"])
    assert not any("exergue" in n for n in zf.namelist())
    dev_name = next(n for n in zf.namelist() if "device" in n)
    dev = np.flipud(np.asarray(Image.open(io.BytesIO(zf.read(dev_name))).convert("L"), dtype=np.int32))
    in_strip = np.zeros((240, 240), dtype=bool)
    in_strip[202:216, 92:148] = True  # inside both the subject block and the strip
    assert (dev[in_strip] < 250).mean() > 0.9, "device lost the subject bottom"

    # Enabled: exergue present as a flat recessed text panel; the device is
    # clipped at the exergue line (the design sits ON the strip, classic
    # coin composition), and the panel is uniformly recessed — no leftover
    # sculpt content.
    plan, zf = run({**base, "exergue_enabled": True})
    assert any("Exergue" in p["label"] for p in plan["passes"])
    ex_name = next(n for n in zf.namelist() if "exergue" in n)
    dev_name = next(n for n in zf.namelist() if "device" in n)
    ex = np.flipud(np.asarray(Image.open(io.BytesIO(zf.read(ex_name))).convert("L"), dtype=np.int32))
    dev = np.flipud(np.asarray(Image.open(io.BytesIO(zf.read(dev_name))).convert("L"), dtype=np.int32))
    assert (dev[in_strip] >= 250).mean() > 0.9, "device not clipped at the exergue line"
    assert (ex[in_strip] < 128).mean() > 0.9, "exergue panel is not a flat recess"
    # Above the strip the device still engraves the subject.
    above = np.zeros((240, 240), dtype=bool)
    above[120:180, 92:148] = True
    assert (dev[above] < 250).mean() > 0.9, "device lost the subject above the strip"


# ----------------------------------------------------------- /render mask_id

def test_render_reuses_wizard_mask_id():
    from fastapi.testclient import TestClient
    from apps.api import service_adapter
    from apps.api.main import app

    client = TestClient(app)

    rng = np.random.default_rng(3)
    photo = rng.uniform(60, 200, (120, 120, 3)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(photo).save(buf, format="PNG")
    up = client.post("/upload", files={"file": ("p.png", buf.getvalue(), "image/png")})
    image_id = up.json()["image_id"]

    hm = np.tile(np.linspace(0, 1, 120, dtype=np.float32), (120, 1))
    buf = io.BytesIO()
    Image.fromarray((hm * 65535).astype(np.uint16)).save(buf, format="PNG")
    hm_path = client.post(
        "/upload/heightmap", files={"file": ("h.png", buf.getvalue(), "image/png")}
    ).json()["heightmap_path"]

    mask = np.zeros((120, 120), dtype=np.float32)
    mask[30:90, 30:90] = 1.0
    mask_id = service_adapter.blob_store.store_heightmap(mask)

    settings = {
        "external_heightmap_path": hm_path,
        "zone_width_mm": 12.0, "zone_height_mm": 12.0,
        "field_pattern": "basket_weave",
    }
    # With mask_id: no "field pattern will engrave over the subject" warning.
    r = client.post("/render", json={
        "image_id": image_id, "settings": settings, "mask_id": mask_id,
    })
    assert r.status_code == 200, r.text
    assert not any("Field pattern" in w for w in r.json()["warnings"])

    # Without a mask: the warning fires.
    r2 = client.post("/render", json={"image_id": image_id, "settings": settings})
    assert r2.status_code == 200, r2.text
    assert any("Field pattern" in w for w in r2.json()["warnings"])

    # Unknown mask_id → 404.
    r3 = client.post("/render", json={
        "image_id": image_id, "settings": settings, "mask_id": "nonexistent",
    })
    assert r3.status_code == 404


def test_all_material_profiles_define_zone_params():
    from pathlib import Path
    import yaml

    profiles_dir = Path(__file__).resolve().parents[1] / "profiles"
    material_profiles = [
        p for p in profiles_dir.glob("mopa_*.yaml")
    ]
    assert len(material_profiles) >= 5
    for p in material_profiles:
        payload = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert payload.get("zone_params"), f"{p.name} lacks zone_params"
        for zone in ("field", "border", "rim", "exergue"):
            assert zone in payload["zone_params"], f"{p.name} zone_params lacks {zone}"
