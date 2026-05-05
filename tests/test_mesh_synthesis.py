"""Tests for the mesh synthesis registry + extruder + STL writer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zoedepth.laser.mesh_input import load_mesh
from zoedepth.laser.mesh_synthesis import (
    DEFAULT_EXTRUDE_BASE_THICKNESS,
    DEFAULT_EXTRUDE_HEIGHT_SCALE,
    DEFAULT_EXTRUDE_SUBSAMPLE,
    DEFAULT_SYNTHESIZER_KEY,
    MAX_EXTRUDE_TRIANGLES_BUDGET,
    MeshSynthesisSpec,
    extrude_heightmap_to_mesh,
    get_synthesizer,
    list_synthesizers,
    load_synthesizer,
    register_synthesizer,
    write_stl_binary,
)


def test_constants_have_documented_values():
    assert DEFAULT_SYNTHESIZER_KEY == "extrude-from-heightmap"
    assert DEFAULT_EXTRUDE_BASE_THICKNESS == 0.05
    assert DEFAULT_EXTRUDE_HEIGHT_SCALE == 0.2
    assert DEFAULT_EXTRUDE_SUBSAMPLE == 1
    assert MAX_EXTRUDE_TRIANGLES_BUDGET == 1_000_000


def test_default_synthesizers_registered():
    keys = {s.key for s in list_synthesizers()}
    assert {"extrude-from-heightmap", "hunyuan3d-2-mini"} <= keys


def test_extrude_is_permissive_hunyuan_is_opt_in():
    assert get_synthesizer("extrude-from-heightmap").requires_opt_in is False
    assert get_synthesizer("hunyuan3d-2-mini").requires_opt_in is True


def test_register_duplicate_raises():
    spec = MeshSynthesisSpec(
        key="extrude-from-heightmap", label="x", license="MIT",
        requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
        loader=lambda d: object(),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_synthesizer(spec)


def test_load_unknown_raises():
    with pytest.raises(KeyError):
        load_synthesizer("not-real", "cpu")


def test_extrude_produces_triangle_soup():
    h = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    mesh = extrude_heightmap_to_mesh(h)
    assert mesh.triangles.ndim == 3
    assert mesh.triangles.shape[1:] == (3, 3)
    assert mesh.triangle_count > 0


def test_extrude_rejects_too_small_heightmap():
    with pytest.raises(ValueError, match="too small"):
        extrude_heightmap_to_mesh(np.zeros((1, 1), dtype=np.float32))


def test_extrude_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        extrude_heightmap_to_mesh(np.zeros((4, 4, 3), dtype=np.float32))


def test_extrude_z_range_matches_height_scale_and_base_thickness():
    h = np.ones((4, 4), dtype=np.float32)
    mesh = extrude_heightmap_to_mesh(h, base_thickness=0.1, height_scale=0.5)
    z = mesh.triangles[..., 2]
    assert z.max() == pytest.approx(0.5, abs=1e-5)
    assert z.min() == pytest.approx(-0.1, abs=1e-5)


def test_extrude_auto_subsamples_to_stay_under_budget():
    # Build a heightmap so large that subsample=1 would exceed the budget,
    # then verify the auto subsampler scales down without raising.
    big = np.zeros((2000, 2000), dtype=np.float32)
    mesh = extrude_heightmap_to_mesh(big, subsample=1)
    assert mesh.triangle_count <= MAX_EXTRUDE_TRIANGLES_BUDGET


def test_default_synth_requires_heightmap():
    synth, _ = load_synthesizer("extrude-from-heightmap", "cpu")
    with pytest.raises(ValueError, match="heightmap"):
        synth.infer(image=None, heightmap=None)


def test_default_synth_round_trips_through_stl_writer(tmp_path: Path):
    synth, _ = load_synthesizer("extrude-from-heightmap", "cpu")
    h = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    mesh = synth.infer(image=None, heightmap=h)
    p = tmp_path / "extrude.stl"
    written = write_stl_binary(mesh, p)
    assert written.exists()
    reloaded = load_mesh(written)
    assert reloaded.triangle_count == mesh.triangle_count


def test_hunyuan_loader_raises_when_package_absent():
    with pytest.raises(RuntimeError, match="Hunyuan3D"):
        load_synthesizer("hunyuan3d-2-mini", "cpu")
