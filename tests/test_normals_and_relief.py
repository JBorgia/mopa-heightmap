"""Tests for :mod:`zoedepth.laser.normals` and the relief composer."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.normals import (
    DEFAULT_PIXEL_SCALE,
    EPS_NORM,
    NormalEstimatorSpec,
    RGB_HALF,
    RGB_MAX,
    depth_to_normals,
    get_estimator,
    list_estimators,
    load_estimator,
    register_estimator,
)
from zoedepth.laser.relief import (
    DEFAULT_BULK_WEIGHT,
    DEFAULT_MICRO_WEIGHT,
    DETAIL_FORM_SLIDER_MAX,
    DETAIL_FORM_SLIDER_MIN,
    EPS_RANGE,
    compose_relief,
    detail_slider_to_weights,
    normalise_unit,
)


# ----------------------------------------------------------- registry

def test_default_normal_estimators_registered_with_correct_licensing():
    fd = get_estimator("finite_diff")
    dsine = get_estimator("dsine")
    marigold = get_estimator("marigold_normals")
    assert fd is not None and not fd.requires_opt_in
    assert dsine is not None and not dsine.requires_opt_in
    assert dsine.license == "Apache-2.0"
    assert marigold is not None and marigold.requires_opt_in
    assert marigold.license.startswith("CC-BY-NC")


def test_list_normal_estimators_filters_opt_in():
    keys_default = {e.key for e in list_estimators(include_opt_in=False)}
    keys_all = {e.key for e in list_estimators(include_opt_in=True)}
    assert "marigold_normals" in keys_all
    assert "marigold_normals" not in keys_default


def test_register_normal_estimator_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register_estimator(NormalEstimatorSpec(
            key="finite_diff", label="dup", license="x",
            requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
            loader=lambda d: None,
        ))


def test_load_unknown_estimator_raises():
    with pytest.raises(KeyError, match="No normal estimator"):
        load_estimator("not_a_real_one", "cpu")


# ----------------------------------------------------------- depth_to_normals

def test_depth_to_normals_flat_surface_points_up():
    z = np.zeros((8, 8), dtype=np.float32)
    n = depth_to_normals(z)
    assert n.shape == (8, 8, 3)
    # All normals should be (0, 0, 1) for a flat surface.
    assert np.allclose(n[..., 0], 0.0)
    assert np.allclose(n[..., 1], 0.0)
    assert np.allclose(n[..., 2], 1.0)


def test_depth_to_normals_unit_length():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(16, 16)).astype(np.float32)
    n = depth_to_normals(z)
    lengths = np.sqrt((n * n).sum(axis=-1))
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_depth_to_normals_ramp_tilts_correctly():
    # z = 0.5 * x — surface tilts right; normal should lean into -x.
    w = 16
    xs = np.arange(w, dtype=np.float32)
    z = np.tile(0.5 * xs, (w, 1))
    n = depth_to_normals(z, pixel_scale=DEFAULT_PIXEL_SCALE)
    interior = n[w // 4 : 3 * w // 4, w // 4 : 3 * w // 4]
    assert interior[..., 0].mean() < -0.1
    assert abs(interior[..., 1].mean()) < 1e-3
    assert interior[..., 2].mean() > 0.5


def test_depth_to_normals_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        depth_to_normals(np.zeros((4, 4, 4), dtype=np.float32))


def test_normals_constants_have_documented_values():
    assert DEFAULT_PIXEL_SCALE == 1.0
    assert EPS_NORM == 1e-8
    assert RGB_HALF == 127.5
    assert RGB_MAX == 255.0


def test_finite_diff_loader_round_trips():
    inst, dev = load_estimator("finite_diff", "cpu")
    z = np.zeros((4, 4), dtype=np.float32)
    n = inst.infer_from_depth(z)
    assert dev == "cpu"
    assert n.shape == (4, 4, 3)


# ----------------------------------------------------------- relief composer

def test_normalise_unit_constant_field_returns_zero():
    arr = np.full((4, 4), 0.42, dtype=np.float32)
    out = normalise_unit(arr)
    assert np.all(out == 0.0)


def test_normalise_unit_rescales_to_unit_range():
    arr = np.array([[-3.0, 1.0], [2.0, 5.0]], dtype=np.float32)
    out = normalise_unit(arr)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_compose_relief_default_weights_match_constants():
    bulk = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    micro = np.zeros((4, 4), dtype=np.float32)
    out = compose_relief(bulk, micro)
    expected = DEFAULT_BULK_WEIGHT * normalise_unit(bulk) + DEFAULT_MICRO_WEIGHT * normalise_unit(micro)
    assert np.allclose(out, np.clip(expected, 0.0, 1.0))


def test_compose_relief_renormalises_arbitrary_weights():
    bulk = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    micro = np.linspace(1.0, 0.0, 16, dtype=np.float32).reshape(4, 4)
    a = compose_relief(bulk, micro, bulk_weight=2.0, micro_weight=2.0)
    b = compose_relief(bulk, micro, bulk_weight=0.5, micro_weight=0.5)
    assert np.allclose(a, b)


def test_compose_relief_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compose_relief(np.zeros((4, 4)), np.zeros((4, 8)))


def test_compose_relief_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="positive"):
        compose_relief(np.zeros((4, 4)), np.zeros((4, 4)),
                       bulk_weight=0.0, micro_weight=0.0)


def test_detail_slider_to_weights_endpoints():
    assert detail_slider_to_weights(DETAIL_FORM_SLIDER_MIN) == (1.0, 0.0)
    assert detail_slider_to_weights(DETAIL_FORM_SLIDER_MAX) == (0.0, 1.0)
    assert detail_slider_to_weights(0.5) == (0.5, 0.5)
    # Out-of-range values clamp.
    assert detail_slider_to_weights(-1.0) == (1.0, 0.0)
    assert detail_slider_to_weights(2.0) == (0.0, 1.0)


def test_relief_constants_have_documented_values():
    assert DEFAULT_BULK_WEIGHT == 0.7
    assert DEFAULT_MICRO_WEIGHT == pytest.approx(0.3)
    assert DETAIL_FORM_SLIDER_MIN == 0.0
    assert DETAIL_FORM_SLIDER_MAX == 1.0
    assert EPS_RANGE == 1e-6
