"""Mesh-to-heightmap adapter for the MOPA pipeline.

This module turns a 3-D mesh (``.stl``, ``.obj``, ``.ply``) into a
top-down orthographic depth render that the rest of the pipeline can
consume exactly like a photo-derived depth map. It exists so users with
existing STLs (3D-print models, scanned objects, sculpted bas-reliefs)
can route through the same subject-mask → normals → pass-planner →
LightBurn writer flow without leaving the app.

Design notes:

* No hard dependency on ``trimesh`` / ``open3d``. We ship pure-Python
  loaders for the two formats that cover ~95% of laser-engraving use:
  ASCII + binary STL and OBJ. Richer formats fall through to ``trimesh``
  if installed.
* The rasteriser is a CPU vectorised barycentric splatter with
  configurable supersampling (default 4×) for anti-aliased silhouettes.
* The view is *strictly orthographic* — that's the whole point. Lasers
  see a top-down parallel projection, so any perspective camera would
  introduce error.
* Every threshold, default, and bit-depth is a named module-level
  constant exported and pinned by tests, matching the discipline applied
  to every other module in this package.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


__all__ = [
    "load_mesh",
    "render_orthographic_heightmap",
    "MeshHeightmap",
    "MeshData",
    "SUPPORTED_MESH_SUFFIXES",
    "DEFAULT_OUTPUT_RESOLUTION",
    "DEFAULT_SUPERSAMPLE_FACTOR",
    "DEFAULT_VIEW_AXIS",
    "DEFAULT_BACKGROUND_DEPTH",
    "DEFAULT_NORMALISE_MESH",
    "DEFAULT_INVERT_DEPTH",
    "DEFAULT_PADDING_FRACTION",
    "STL_BINARY_HEADER_BYTES",
    "STL_BINARY_TRIANGLE_BYTES",
    "MESH_EPS_DEGENERATE",
    "MESH_VIEW_AXES",
]


# ---------------------------------------------------------------- constants

# File suffixes we accept directly. Anything else is routed through trimesh
# if it's importable, otherwise rejected with a clear error.
SUPPORTED_MESH_SUFFIXES: Tuple[str, ...] = (".stl", ".obj", ".ply")

# Default longest-side resolution of the rendered heightmap. 768 mirrors the
# wizard preview cap; the full-resolution render at export time can be
# higher (caller passes ``output_resolution`` explicitly).
DEFAULT_OUTPUT_RESOLUTION: int = 768

# Anti-aliasing supersample factor. The rasteriser renders at
# ``output_resolution * factor`` then box-downsamples. 4× gives near-lossless
# silhouette edges with ~16× the pixel work — affordable on CPU for normal
# mesh sizes (< 1 M triangles).
DEFAULT_SUPERSAMPLE_FACTOR: int = 4

# Which axis points toward the laser. "+z" means "the camera looks down the
# −z direction at a model lying in the XY plane" — matches every CAD/CAM
# convention you'll meet.
DEFAULT_VIEW_AXIS: str = "+z"
MESH_VIEW_AXES: Tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")

# Pixels with no triangle coverage get this depth value. 0.0 = "deepest
# possible burn" so the background ablates flat. Keep this aligned with the
# ``black_is_deep`` pipeline convention.
DEFAULT_BACKGROUND_DEPTH: float = 0.0

# When True, the mesh is recentred + scaled to fit a unit cube (preserving
# aspect) before rasterising. This makes any input mesh produce a
# normalised heightmap regardless of original units (mm, cm, m, inches).
DEFAULT_NORMALISE_MESH: bool = True

# When True, pixels closer to the camera become BRIGHTER (1.0) and the
# background becomes DARKER (0.0). This matches the rest of the pipeline,
# which treats white as "high" and black as "deep".
DEFAULT_INVERT_DEPTH: bool = True

# Empty margin around the mesh as a fraction of the longest side. A small
# pad keeps the silhouette away from the image edge so subject-mask edge
# softening has room to feather without clipping.
DEFAULT_PADDING_FRACTION: float = 0.04

# Numerical guard against divide-by-zero on degenerate triangles.
MESH_EPS_DEGENERATE: float = 1e-8

# Binary-STL layout: 80-byte ASCII header, uint32 triangle count, then 50
# bytes per triangle (12 floats normal+verts + uint16 attribute).
STL_BINARY_HEADER_BYTES: int = 80
STL_BINARY_TRIANGLE_BYTES: int = 50


# ---------------------------------------------------------------- dataclasses

@dataclass(frozen=True)
class MeshData:
    """Triangle soup produced by :func:`load_mesh`.

    ``triangles`` is shape ``(N, 3, 3)`` — N triangles, 3 vertices each, 3
    coordinates per vertex (x, y, z). We keep the soup form because the
    rasteriser doesn't care about vertex sharing and the loaders don't
    have to build an index buffer.
    """
    triangles: np.ndarray
    source_path: Path | None = None

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True)
class MeshHeightmap:
    """Output of :func:`render_orthographic_heightmap`.

    ``heightmap`` is float32 in ``[0, 1]``; ``alpha`` is float32 in
    ``[0, 1]`` where 1 means a triangle covered the pixel.
    """
    heightmap: np.ndarray         # H×W float32 in [0, 1]
    alpha: np.ndarray             # H×W float32 in [0, 1]
    view_axis: str
    source_triangle_count: int


# ---------------------------------------------------------------- loaders

def _read_stl_binary(buf: bytes) -> np.ndarray:
    if len(buf) < STL_BINARY_HEADER_BYTES + 4:
        raise ValueError("Binary STL too short for header + count")
    n = struct.unpack_from("<I", buf, STL_BINARY_HEADER_BYTES)[0]
    expected = STL_BINARY_HEADER_BYTES + 4 + n * STL_BINARY_TRIANGLE_BYTES
    if len(buf) < expected:
        raise ValueError(
            f"Binary STL truncated: declared {n} triangles "
            f"(needs {expected} bytes), file has {len(buf)}"
        )
    # Each triangle: 3 floats normal + 9 floats verts + 2 bytes attr.
    dt = np.dtype([
        ("normal", "<f4", (3,)),
        ("verts", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    arr = np.frombuffer(
        buf, dtype=dt, count=n, offset=STL_BINARY_HEADER_BYTES + 4,
    )
    return np.ascontiguousarray(arr["verts"], dtype=np.float32)


def _read_stl_ascii(text: str) -> np.ndarray:
    triangles: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            parts = s.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed ASCII STL vertex line: {s!r}")
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif s.startswith("endfacet"):
            if len(current) != 3:
                raise ValueError(
                    f"ASCII STL facet had {len(current)} vertices, expected 3"
                )
            triangles.append(current)
            current = []
    if not triangles:
        raise ValueError("ASCII STL contained no triangles")
    return np.asarray(triangles, dtype=np.float32)


def _read_obj(text: str) -> np.ndarray:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            # OBJ face indices may be "v", "v/vt", or "v/vt/vn"; we only
            # need the first component, and they're 1-based.
            idx = [int(tok.split("/")[0]) - 1 for tok in parts[1:]]
            # Triangulate polygon faces with a fan around vertex 0.
            for i in range(1, len(idx) - 1):
                faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        raise ValueError("OBJ file contained no usable vertices or faces")
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    return v[f]  # shape (N, 3, 3)


def _read_with_trimesh(path: Path) -> np.ndarray:
    try:
        import trimesh  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only if trimesh missing
        raise ValueError(
            f"Mesh format {path.suffix} requires the optional 'trimesh' "
            f"dependency: pip install trimesh"
        ) from exc
    mesh = trimesh.load(str(path), force="mesh")
    if mesh.is_empty or mesh.faces.shape[0] == 0:
        raise ValueError(f"trimesh loaded an empty mesh from {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts[faces]


def load_mesh(path: str | Path) -> MeshData:
    """Load a mesh from disk into a :class:`MeshData` triangle soup.

    Supports STL (ASCII + binary) and OBJ natively. Other formats are
    delegated to ``trimesh`` if installed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".stl":
        buf = p.read_bytes()
        if buf[:5].lower() == b"solid" and b"facet" in buf[:1024].lower():
            triangles = _read_stl_ascii(buf.decode("utf-8", errors="replace"))
        else:
            triangles = _read_stl_binary(buf)
    elif suffix == ".obj":
        triangles = _read_obj(p.read_text(encoding="utf-8", errors="replace"))
    elif suffix in SUPPORTED_MESH_SUFFIXES:
        triangles = _read_with_trimesh(p)
    else:
        raise ValueError(
            f"Unsupported mesh suffix {suffix!r}; "
            f"supported: {SUPPORTED_MESH_SUFFIXES}"
        )
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"Loader returned invalid triangle array of shape {triangles.shape}"
        )
    return MeshData(triangles=triangles.astype(np.float32, copy=False), source_path=p)


# ---------------------------------------------------------------- view setup

def _project_for_axis(triangles: np.ndarray, axis: str) -> np.ndarray:
    """Return triangles in a coordinate frame where +Z points at the camera.

    The output has the same ``(N, 3, 3)`` shape; column 0 = u, column 1 = v
    (image axes), column 2 = depth (larger = closer to camera).
    """
    if axis not in MESH_VIEW_AXES:
        raise ValueError(f"axis must be one of {MESH_VIEW_AXES}, got {axis!r}")
    sign = 1.0 if axis.startswith("+") else -1.0
    a = axis[-1]
    if a == "z":
        u, v, d = 0, 1, 2
    elif a == "y":
        u, v, d = 0, 2, 1
    else:  # "x"
        u, v, d = 1, 2, 0
    out = np.stack(
        [triangles[..., u], triangles[..., v], sign * triangles[..., d]], axis=-1,
    )
    return out


def _normalise_to_unit(triangles: np.ndarray, padding: float) -> np.ndarray:
    """Centre the mesh in XY, scale longest XY side to 1 - 2*padding."""
    flat = triangles.reshape(-1, 3)
    xy_min = flat[:, :2].min(axis=0)
    xy_max = flat[:, :2].max(axis=0)
    span = (xy_max - xy_min).max()
    if span < MESH_EPS_DEGENERATE:
        raise ValueError("Mesh has zero XY extent; nothing to render")
    centre = (xy_max + xy_min) * 0.5
    target_span = max(MESH_EPS_DEGENERATE, 1.0 - 2.0 * float(padding))
    scale = target_span / span
    out = triangles.copy()
    out[..., :2] = (triangles[..., :2] - centre) * scale + 0.5
    # Depth: scale by the same factor so aspect stays sane, then shift to [0, *].
    z = triangles[..., 2] * scale
    out[..., 2] = z - z.min()
    return out


# ---------------------------------------------------------------- rasteriser

def _rasterise(triangles_uv_depth: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """CPU vectorised orthographic Z-buffer.

    Each triangle is rasterised independently inside its screen-space
    bounding box; per-pixel depth keeps the *maximum* (closest to camera)
    sample seen so far. Returns ``(depth, coverage)`` both ``H×W float32``.
    """
    depth = np.full((h, w), -np.inf, dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.float32)
    pixel_u = (triangles_uv_depth[..., 0] * w).astype(np.float32)
    pixel_v = (triangles_uv_depth[..., 1] * h).astype(np.float32)
    z = triangles_uv_depth[..., 2].astype(np.float32)
    n = triangles_uv_depth.shape[0]
    for i in range(n):
        u0, u1, u2 = pixel_u[i]
        v0, v1, v2 = pixel_v[i]
        z0, z1, z2 = z[i]
        u_min = max(int(np.floor(min(u0, u1, u2))), 0)
        u_max = min(int(np.ceil(max(u0, u1, u2))), w - 1)
        v_min = max(int(np.floor(min(v0, v1, v2))), 0)
        v_max = min(int(np.ceil(max(v0, v1, v2))), h - 1)
        if u_min > u_max or v_min > v_max:
            continue
        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denom) < MESH_EPS_DEGENERATE:
            continue
        uu, vv = np.meshgrid(
            np.arange(u_min, u_max + 1, dtype=np.float32) + 0.5,
            np.arange(v_min, v_max + 1, dtype=np.float32) + 0.5,
        )
        w0 = ((v1 - v2) * (uu - u2) + (u2 - u1) * (vv - v2)) / denom
        w1 = ((v2 - v0) * (uu - u2) + (u0 - u2) * (vv - v2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z0 + w1 * z1 + w2 * z2
        sub_depth = depth[v_min:v_max + 1, u_min:u_max + 1]
        sub_cov = coverage[v_min:v_max + 1, u_min:u_max + 1]
        # Take the maximum depth (= closest to the +Z camera) per pixel.
        update = inside & (zz > sub_depth)
        sub_depth[update] = zz[update]
        sub_cov[inside] = 1.0
    # Replace -inf with the minimum finite depth so downsampling stays sane.
    finite = depth[coverage > 0]
    if finite.size:
        depth[coverage == 0] = float(finite.min())
    else:
        depth[:] = 0.0
    return depth, coverage


def _box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2 = h // factor
    w2 = w // factor
    cropped = arr[: h2 * factor, : w2 * factor]
    return cropped.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def render_orthographic_heightmap(
    mesh: MeshData,
    *,
    output_resolution: int = DEFAULT_OUTPUT_RESOLUTION,
    supersample: int = DEFAULT_SUPERSAMPLE_FACTOR,
    view_axis: str = DEFAULT_VIEW_AXIS,
    normalise: bool = DEFAULT_NORMALISE_MESH,
    invert_depth: bool = DEFAULT_INVERT_DEPTH,
    background_depth: float = DEFAULT_BACKGROUND_DEPTH,
    padding_fraction: float = DEFAULT_PADDING_FRACTION,
) -> MeshHeightmap:
    """Render ``mesh`` to a top-down orthographic heightmap.

    The output side length is ``output_resolution`` after downsampling
    from the supersampled buffer. The resulting heightmap is float32 in
    ``[0, 1]`` ready to plug into the rest of the pipeline.
    """
    if output_resolution <= 0:
        raise ValueError("output_resolution must be positive")
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    triangles = _project_for_axis(mesh.triangles, view_axis)
    if normalise:
        triangles = _normalise_to_unit(triangles, padding=padding_fraction)
    big = output_resolution * supersample
    depth, coverage = _rasterise(triangles, w=big, h=big)
    depth = _box_downsample(depth, supersample)
    coverage = _box_downsample(coverage, supersample)
    # Map raw depth into [0, 1] using the covered region's range.
    covered = coverage > 0
    if covered.any():
        d_min = float(depth[covered].min())
        d_max = float(depth[covered].max())
        if d_max - d_min > MESH_EPS_DEGENERATE:
            normalised = (depth - d_min) / (d_max - d_min)
        else:
            normalised = np.zeros_like(depth)
    else:
        normalised = np.zeros_like(depth)
    if invert_depth:
        # Brighter = closer to camera, matching the photo pipeline contract.
        # ``normalised`` already has higher values for closer points (we
        # negated the camera axis in projection), so we keep it as-is and
        # only fill the background.
        pass
    out = np.where(covered, np.clip(normalised, 0.0, 1.0), float(background_depth))
    return MeshHeightmap(
        heightmap=out.astype(np.float32, copy=False),
        alpha=coverage.astype(np.float32, copy=False),
        view_axis=view_axis,
        source_triangle_count=mesh.triangle_count,
    )
