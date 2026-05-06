"""Tests for the click-driven mask registry and flood-fill backend."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mopa.click_mask import (
    DEFAULT_CLICKER_KEY,
    DEFAULT_FLOOD_MAX_FRACTION,
    DEFAULT_FLOOD_TOLERANCE,
    POINT_LABEL_NEGATIVE,
    POINT_LABEL_POSITIVE,
    ClickMaskerSpec,
    get_clicker,
    list_clickers,
    load_clicker,
    register_clicker,
)


def test_constants_have_documented_values():
    assert DEFAULT_CLICKER_KEY == "flood-fill"
    assert DEFAULT_FLOOD_TOLERANCE == 0.08
    assert DEFAULT_FLOOD_MAX_FRACTION == 0.6
    assert POINT_LABEL_POSITIVE == 1
    assert POINT_LABEL_NEGATIVE == 0


def test_default_clickers_registered():
    keys = {s.key for s in list_clickers()}
    assert {"flood-fill", "sam2-tiny"} <= keys


def test_flood_fill_is_permissive_sam2_is_opt_in():
    assert get_clicker("flood-fill").requires_opt_in is False
    assert get_clicker("sam2-tiny").requires_opt_in is True


def test_register_duplicate_raises():
    spec = ClickMaskerSpec(
        key="flood-fill", label="x", license="MIT",
        requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
        loader=lambda d: object(),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_clicker(spec)


def test_load_unknown_raises():
    with pytest.raises(KeyError):
        load_clicker("not-real", "cpu")


def _two_square_image() -> Image.Image:
    arr = np.zeros((40, 80), dtype=np.uint8)
    arr[10:30, 10:30] = 200    # left bright square
    arr[10:30, 50:70] = 100    # right medium square
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_flood_fill_grows_from_positive_click():
    clicker, _ = load_clicker("flood-fill", "cpu")
    img = _two_square_image()
    alpha = clicker.infer(img, [(20, 20)], [POINT_LABEL_POSITIVE])
    assert alpha.shape == (40, 80)
    # Centre of left square is included; centre of right square is not.
    assert alpha[20, 20] == 1.0
    assert alpha[20, 60] == 0.0
    # Background pixel is not included.
    assert alpha[5, 5] == 0.0


def test_flood_fill_negative_click_subtracts():
    clicker, _ = load_clicker("flood-fill", "cpu")
    img = _two_square_image()
    alpha = clicker.infer(
        img,
        [(20, 20), (60, 20)],
        [POINT_LABEL_POSITIVE, POINT_LABEL_NEGATIVE],
    )
    assert alpha[20, 20] == 1.0
    assert alpha[20, 60] == 0.0


def test_flood_fill_ignores_out_of_bounds_clicks():
    clicker, _ = load_clicker("flood-fill", "cpu")
    img = _two_square_image()
    alpha = clicker.infer(img, [(-1, -1), (99999, 99999)], [POINT_LABEL_POSITIVE] * 2)
    assert alpha.sum() == 0.0


def test_flood_fill_no_clicks_returns_empty_alpha():
    clicker, _ = load_clicker("flood-fill", "cpu")
    img = _two_square_image()
    alpha = clicker.infer(img, [], [])
    assert alpha.shape == (40, 80)
    assert alpha.sum() == 0.0


def test_sam2_loader_raises_when_package_absent():
    with pytest.raises(RuntimeError, match="SAM-2"):
        load_clicker("sam2-tiny", "cpu")
