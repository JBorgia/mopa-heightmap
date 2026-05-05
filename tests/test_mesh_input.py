"""Tests for the mesh-input adapter (Phase 7)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from zoedepth.laser.mesh_input import (
    DEFAULT_BACKGROUND_DEPTH,
    DEFAULT_INVERT_DEPTH,
    DEFAULT_NORMALISE_MESH,
    DEFAULT_OUTPUT_RESOLUTION,
    DEFAULT_PADDING_FRACTION,
    DEFAULT_SUPERSAMPLE_FACTOR,
    DEFAULT_VIEW_AXIS,
    MESH_EPS_DEGENERATE,
    MESH_VIEW_AXES,
    STL_BINARY_HEADER_BYTES,
    STL_BINARY_TRIANGLE_BYTES,
    SUPPORTED_MESH_SUFFIXES,
    MeshData,
    MeshHeightmap,
    load_mesh,
    render_orthographic_heightmap,
)


# ----------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert SUPPORTED_MESH_SUFFIXES == (".stl", ".obj", ".ply")
    assert DEFAULT_OUTPUT_RESOLUTION == 768
    assert DEFAULT_SUPERSAMPLE_FACTOR == 4
    assert DEFAULT_VIEW_AXIS == "+z"
    assert DEFAULT_BACKGROUND_DEPTH == 0.0
    assert DEFAULT_NORMALISE_MESH is True
    assert DEFAULT_INVERT_DEPTH is True
    assert DEFAULT_PADDING_FRACTION == 0.04
    assert STL_BINARY_HEADER_BYTES == 80
    assert STL_BINARY_TRIANGLE_BYTES == 50
    assert MESH_EPS_DEGENERATE == 1e-8
    assert set(MESH_VIEW_AXES) == {"+x", "-x", "+y", "-y", "+z", "-z"}


# ----------------------------------------------------------- helpers

def _write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(b"\x00" * STL_BINARY_HEADER_BYTES)
        f.write(struct.pack("<I", triangles.shape[0]))
        for tri in triangles:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))      # normal
            for v in tri:
                f.write(struct.pack("<3f", *v.astype(float)))
            f.write(struct.pack("<H", 0))                    # attribute


def _write_ascii_stl(path: Path, triangles: np.ndarray) -> None:
    lines = ["solid test"]
    for tri in triangles:
        lines.append(" facet normal 0 0 0")
        lines.append("  outer loop")
        for v in tri:
            lines.append(f"   vertex {v[0]} {v[1]} {v[2]}")
        lines.append("  endloop")
        lines.append(" endfacet")
    lines.append("endsolid test")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _pyramid_triangles() -> np.ndarray:
    """Square-base pyramid centred at origin, apex at +z."""
    apex = np.array([0.0, 0.0, 1.0])
    a = np.array([-1.0, -1.0, 0.0])
    b = np.array([+1.0, -1.0, 0.0])
    c = np.array([+1.0, +1.0, 0.0])
    d = np.array([-1.0, +1.0, 0.0])
    return np.array([
        [a, b, apex],
        [b, c, apex],
        [c, d, apex],
        [d, a, apex],
        # Base (so the lower silhouette has full coverage):
        [a, b, c],
        [a, c, d],
    ], dtype=np.float32)


# ----------------------------------------------------------- loaders

def test_load_binary_stl_round_trips_triangle_count(tmp_path: Path):
    tris = _pyramid_triangles()
    p = tmp_path / "pyramid.stl"
    _write_binary_stl(p, tris)
    mesh = load_mesh(p)
    assert isinstance(mesh, MeshData)
    assert mesh.triangle_count == tris.shape[0]
    assert mesh.triangles.shape == tris.shape
    assert np.allclose(mesh.triangles, tris)


def test_load_ascii_stl_round_trips_triangle_count(tmp_path: Path):
    tris = _pyramid_triangles()
    p = tmp_path / "pyramid_ascii.stl"
    _write_ascii_stl(p, tris)
    mesh = load_mesh(p)
    assert mesh.triangle_count == tris.shape[0]
    assert np.allclose(mesh.triangles, tris)


def test_load_obj(tmp_path: Path):
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    p = tmp_path / "tetra.obj"
    _write_obj(p, vertices, faces)
    mesh = load_mesh(p)
    assert mesh.triangle_count == 4


def test_load_obj_triangulates_quad(tmp_path: Path):
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    p = tmp_path / "quad.obj"
    p.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="utf-8"
    )
    mesh = load_mesh(p)
    assert mesh.triangle_count == 2  # fan-triangulated


def test_load_mesh_rejects_unknown_suffix(tmp_path: Path):
    p = tmp_path / "model.xyz"
    p.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        load_mesh(p)


def test_load_mesh_missing_file():
    with pytest.raises(FileNotFoundError):
        load_mesh(Path("does_not_exist.stl"))


def test_load_ascii_stl_rejects_malformed_facet(tmp_path: Path):
    p = tmp_path / "bad.stl"
    p.write_text(
        "solid x\n facet normal 0 0 0\n  outer loop\n"
        "   vertex 0 0 0\n   vertex 1 0 0\n  endloop\n endfacet\nendsolid x"
    )
    with pytest.raises(ValueError, match="3"):
        load_mesh(p)


# ----------------------------------------------------------- rasteriser

def test_render_returns_normalised_heightmap_in_unit_range(tmp_path: Path):
    tris = _pyramid_triangles()
    mesh = MeshData(triangles=tris)
    out = render_orthographic_heightmap(
        mesh, output_resolution=64, supersample=2,
    )
    assert isinstance(out, MeshHeightmap)
    assert out.heightmap.dtype == np.float32
    assert out.heightmap.shape == (64, 64)
    assert out.heightmap.min() >= 0.0
    assert out.heightmap.max() <= 1.0


def test_render_pyramid_apex_is_brightest_pixel():
    tris = _pyramid_triangles()
    mesh = MeshData(triangles=tris)
    out = render_orthographic_heightmap(
        mesh, output_resolution=64, supersample=2,
    )
    cy, cx = np.unravel_index(int(np.argmax(out.heightmap)), out.heightmap.shape)
    # Apex of a centred pyramid projects to the image centre (within a few px).
    assert abs(cy - 32) <= 4
    assert abs(cx - 32) <= 4
    assert out.heightmap[cy, cx] == pytest.approx(1.0, abs=1e-3)


def test_render_alpha_marks_subject_coverage():
    tris = _pyramid_triangles()
    mesh = MeshData(triangles=tris)
    out = render_orthographic_heightmap(
        mesh, output_resolution=64, supersample=2,
    )
    # Pyramid base covers the whole image (top-down view); coverage near 1.
    assert float(out.alpha.mean()) > 0.5
    assert out.source_triangle_count == tris.shape[0]


def test_render_view_axis_changes_silhouette():
    tris = _pyramid_triangles()
    mesh = MeshData(triangles=tris)
    top = render_orthographic_heightmap(
        mesh, output_resolution=48, supersample=1, view_axis="+z",
    )
    side = render_orthographic_heightmap(
        mesh, output_resolution=48, supersample=1, view_axis="+y",
    )
    # The two projections must differ — if they don't, the axis switch is broken.
    assert not np.allclose(top.heightmap, side.heightmap)


def test_render_rejects_unknown_axis():
    mesh = MeshData(triangles=_pyramid_triangles())
    with pytest.raises(ValueError, match="axis"):
        render_orthographic_heightmap(mesh, view_axis="diagonal")


def test_render_rejects_invalid_resolution():
    mesh = MeshData(triangles=_pyramid_triangles())
    with pytest.raises(ValueError, match="output_resolution"):
        render_orthographic_heightmap(mesh, output_resolution=0)
    with pytest.raises(ValueError, match="supersample"):
        render_orthographic_heightmap(mesh, supersample=0)


def test_render_handles_zero_extent_mesh():
    tris = np.zeros((1, 3, 3), dtype=np.float32)
    mesh = MeshData(triangles=tris)
    with pytest.raises(ValueError, match="zero XY extent"):
        render_orthographic_heightmap(mesh, output_resolution=16, supersample=1)


def test_render_background_value_is_applied():
    tris = _pyramid_triangles()
    mesh = MeshData(triangles=tris)
    bg = 0.25
    out = render_orthographic_heightmap(
        mesh, output_resolution=64, supersample=1,
        background_depth=bg, padding_fraction=0.2,
    )
    # The padded border should equal the background value.
    assert out.heightmap[0, 0] == pytest.approx(bg)
    assert out.alpha[0, 0] == 0.0


def test_render_full_pipeline_load_and_render(tmp_path: Path):
    tris = _pyramid_triangles()
    p = tmp_path / "pyramid.stl"
    _write_binary_stl(p, tris)
    mesh = load_mesh(p)
    out = render_orthographic_heightmap(mesh, output_resolution=32, supersample=2)
    assert out.heightmap.shape == (32, 32)
    assert out.source_triangle_count == tris.shape[0]
