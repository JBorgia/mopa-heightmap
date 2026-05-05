import numpy as np
import pytest
from PIL import Image

from zoedepth.laser.service import (
    DEFAULT_SETTINGS,
    ExportRequest,
    HeightmapService,
    merge_profile_settings,
)
from zoedepth.laser.settings import AppSettings


def _write_synthetic_heightmap(target, w=48, h=48):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (max(w, h) * 0.18) ** 2))
    arr = (0.3 + 0.7 * bump).astype(np.float32)
    arr16 = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr16, mode="I;16").save(target)
    return target


def test_merge_profile_settings_overrides_take_precedence():
    profile = {"black_is_deep": False, "heightmap": {"polarity_invert": True}}
    s = merge_profile_settings(profile, {"polarity_invert": False, "background_value": None})
    assert s["polarity_invert"] is False
    assert s["black_is_deep"] is False
    # None overrides do not clobber profile defaults.
    assert s["background_value"] == DEFAULT_SETTINGS["background_value"]


def test_render_requires_external_heightmap_path(tmp_path):
    svc = HeightmapService(app_settings=AppSettings())
    img = Image.new("RGB", (32, 32), color=(120, 120, 120))
    settings = merge_profile_settings(None, None)
    with pytest.raises(ValueError, match="external heightmap"):
        svc.render(img, settings)


def test_render_returns_passthrough_heightmap(tmp_path):
    svc = HeightmapService(app_settings=AppSettings())
    img = Image.new("RGB", (48, 48), color=(80, 90, 100))
    heightmap_path = _write_synthetic_heightmap(tmp_path / "fixture.png")
    settings = merge_profile_settings(
        None, {"external_heightmap_path": str(heightmap_path)},
    )

    result = svc.render(img, settings)
    assert result.heightmap.shape == (48, 48)
    assert result.heightmap.dtype == np.float32
    # Sculptok bright_raised: center bump should be > corners.
    h, w = result.heightmap.shape
    assert result.heightmap[h // 2, w // 2] > result.heightmap[0, 0]


def test_render_polarity_invert_flips_heightmap(tmp_path):
    svc = HeightmapService(app_settings=AppSettings())
    img = Image.new("RGB", (48, 48), color=(80, 90, 100))
    heightmap_path = _write_synthetic_heightmap(tmp_path / "fixture.png")

    settings_normal = merge_profile_settings(
        None, {"external_heightmap_path": str(heightmap_path)},
    )
    settings_invert = merge_profile_settings(
        None,
        {
            "external_heightmap_path": str(heightmap_path),
            "polarity_invert": True,
        },
    )
    normal = svc.render(img, settings_normal).heightmap
    inverted = svc.render(img, settings_invert).heightmap
    assert np.allclose(inverted, 1.0 - normal, atol=1e-5)


def test_service_export_writes_full_bundle(tmp_path):
    svc = HeightmapService(app_settings=AppSettings())
    img = Image.new("RGB", (48, 48), color=(80, 90, 100))
    heightmap_path = _write_synthetic_heightmap(tmp_path / "fixture.png")
    settings = merge_profile_settings(
        None, {"external_heightmap_path": str(heightmap_path)},
    )
    request = ExportRequest(
        output_dir=tmp_path / "out",
        base_stem="thing",
        write_preview=True,
        write_calibration_ramp=True,
    )
    bundle = svc.export(img, settings, request)
    assert bundle.lightburn_png.exists()
    assert bundle.master16_png.exists()
    assert bundle.preview_png and bundle.preview_png.exists()
    assert bundle.ramp_png and bundle.ramp_png.exists()
    assert bundle.settings_json.exists()
