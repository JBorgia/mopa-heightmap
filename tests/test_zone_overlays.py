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
