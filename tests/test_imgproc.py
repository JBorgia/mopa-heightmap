"""Tests for Stage A input conditioning."""
from __future__ import annotations

import numpy as np
from PIL import Image

from zoedepth.laser.imgproc import (
    InputConditioningSettings,
    cap_longest_side,
    clahe_lightness,
    condition_input,
    denoise_nlm,
    gray_world_white_balance,
    remove_specular_highlights,
    settings_from_mapping,
)


def _solid(size=(64, 48), color=(120, 80, 60)) -> Image.Image:
    return Image.new("RGB", size, color)


def _gradient(size=(64, 48)) -> Image.Image:
    arr = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1], 1))
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb)


def test_condition_input_default_is_orient_only():
    img = _solid()
    out = condition_input(img)
    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert out.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(out), np.asarray(img))


def test_white_balance_neutralizes_global_cast():
    img = _solid(color=(200, 100, 50))
    out = gray_world_white_balance(img)
    arr = np.asarray(out, dtype=np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    assert np.allclose(means, means.mean(), atol=1.0)


def test_clahe_changes_pixels_on_gradient():
    img = _gradient()
    out = clahe_lightness(img, clip_limit=4.0, tile_grid=4)
    assert not np.array_equal(np.asarray(img), np.asarray(out))


def test_denoise_runs_and_preserves_shape():
    img = _gradient()
    out = denoise_nlm(img, strength=3.0)
    assert out.size == img.size
    assert out.mode == "RGB"


def test_remove_specular_no_op_when_no_highlights():
    img = _solid(color=(100, 100, 100))
    out = remove_specular_highlights(img, threshold=240)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(img))


def test_remove_specular_inpaints_bright_patch():
    arr = np.full((40, 40, 3), 100, dtype=np.uint8)
    arr[10:20, 10:20] = 255
    img = Image.fromarray(arr)
    out = np.asarray(remove_specular_highlights(img, threshold=240, inpaint_radius=3))
    # The bright patch should be brought down toward background.
    assert out[15, 15].max() < 250


def test_cap_longest_side_downscales():
    img = _solid(size=(800, 400))
    out = cap_longest_side(img, max_dim=200)
    assert max(out.size) == 200
    # Aspect ratio preserved.
    assert out.size == (200, 100)


def test_cap_longest_side_no_op_under_limit():
    img = _solid(size=(80, 40))
    out = cap_longest_side(img, max_dim=200)
    assert out.size == img.size


def test_settings_from_mapping_overrides_defaults():
    cfg = settings_from_mapping({"clahe": True, "clahe_clip": 3.5, "denoise_strength": 8.0})
    assert isinstance(cfg, InputConditioningSettings)
    assert cfg.clahe is True
    assert cfg.clahe_clip == 3.5
    assert cfg.denoise_strength == 8.0
    assert cfg.auto_orient is True  # default preserved


def test_condition_input_pipeline_runs_all_toggles():
    img = _gradient()
    cfg = InputConditioningSettings(
        white_balance=True, clahe=True, denoise=True,
        remove_specular=True, max_input_dim=32,
    )
    out = condition_input(img, cfg)
    assert max(out.size) == 32
