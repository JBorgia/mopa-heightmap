"""Image-to-mesh synthesis backends.

Default backend: ``extrude-from-heightmap`` — given a heightmap (the
output of our existing pipeline) produce a printable / CAM-able mesh by
triangulating the regular grid into the top surface, plus a flat base
and four side walls. Pure NumPy.

Opt-in backend: ``hunyuan3d-2-mini`` — Tencent's 0.6 B image→3D model
(non-commercial). Stubbed; requires ``hunyuan3d`` package + 6 GB VRAM
minimum.

The default backend gives users a portable ``.stl`` even without ML
dependencies, and lets us downstream into the existing
:mod:`zoedepth.laser.mesh_input` rasteriser for QC.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .mesh_input import MeshData


__all__ = [
    "MeshSynthesisSpec",
    "register_synthesizer",
    "get_synthesizer",
    "list_synthesizers",
    "load_synthesizer",
    "extrude_heightmap_to_mesh",
    "write_stl_binary",
    "DEFAULT_SYNTHESIZER_KEY",
    "DEFAULT_EXTRUDE_BASE_THICKNESS",
    "DEFAULT_EXTRUDE_HEIGHT_SCALE",
    "DEFAULT_EXTRUDE_SUBSAMPLE",
    "MAX_EXTRUDE_TRIANGLES_BUDGET",
]


# ----------------------------------------------------------- constants

DEFAULT_SYNTHESIZER_KEY: str = "extrude-from-heightmap"

# Thickness of the flat base under the extruded surface, in normalised
# mesh units (the mesh is unit-cube-normalised by convention so 0.05 = 5%
# of the longest dimension).
DEFAULT_EXTRUDE_BASE_THICKNESS: float = 0.05

# Vertical scale applied to the heightmap when extruding. 0.2 keeps
# bas-relief proportions believable; raise toward 1.0 for sculpture-like
# depth.
DEFAULT_EXTRUDE_HEIGHT_SCALE: float = 0.2

# Take every Nth pixel when extruding to keep triangle count manageable.
# 1 = full resolution (use only for small heightmaps).
DEFAULT_EXTRUDE_SUBSAMPLE: int = 1

# Soft cap on triangle count; the extruder auto-raises subsample if the
# heightmap is large enough that full resolution would exceed this.
MAX_EXTRUDE_TRIANGLES_BUDGET: int = 1_000_000

# STL binary record sizes (mirror of mesh_input constants — duplicated
# here so this module is self-contained for write_stl_binary).
_STL_HEADER_BYTES: int = 80


# ----------------------------------------------------------- registry

@dataclass(frozen=True)
class MeshSynthesisSpec:
    """Metadata + loader for one image-to-mesh backend."""

    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]      # device -> object with .infer(image, heightmap=None)


_REGISTRY: Dict[str, MeshSynthesisSpec] = {}


def register_synthesizer(spec: MeshSynthesisSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Mesh synthesizer already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_synthesizer(key: str) -> MeshSynthesisSpec | None:
    return _REGISTRY.get(key)


def list_synthesizers(include_opt_in: bool = True) -> Tuple[MeshSynthesisSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_synthesizer(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No mesh synthesizer registered for: {key!r}")
    return spec.loader(device), device


# ----------------------------------------------------------- extruder

def _auto_subsample(h: int, w: int, requested: int) -> int:
    """Pick a subsample factor that keeps total triangle count under budget."""
    s = max(1, int(requested))
    while True:
        rows = max(2, h // s)
        cols = max(2, w // s)
        # Top: 2 tris/cell × (rows-1)(cols-1). Sides: 4 walls of similar size.
        # Base: 2 tris.
        tris = 2 * (rows - 1) * (cols - 1) + 4 * 2 * (max(rows, cols) - 1) + 2
        if tris <= MAX_EXTRUDE_TRIANGLES_BUDGET:
            return s
        s += 1


def extrude_heightmap_to_mesh(
    heightmap: np.ndarray,
    *,
    base_thickness: float = DEFAULT_EXTRUDE_BASE_THICKNESS,
    height_scale: float = DEFAULT_EXTRUDE_HEIGHT_SCALE,
    subsample: int = DEFAULT_EXTRUDE_SUBSAMPLE,
) -> MeshData:
    """Turn a 2-D heightmap into a closed manifold mesh as triangle soup.

    Coordinates are normalised: x, y span ``[0, 1]`` over the heightmap
    extent (preserving aspect), z spans ``[-base_thickness, height_scale]``
    so the base sits below the XY plane.
    """
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must be 2-D, got shape {heightmap.shape}")
    h_in, w_in = heightmap.shape
    if h_in < 2 or w_in < 2:
        raise ValueError(f"heightmap too small to extrude: {heightmap.shape}")
    s = _auto_subsample(h_in, w_in, subsample)
    sampled = heightmap[::s, ::s].astype(np.float32, copy=False)
    h, w = sampled.shape
    aspect = w_in / float(h_in)
    # Build vertex grid (h × w × 3): top surface.
    xs = np.linspace(0.0, aspect, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.clip(sampled, 0.0, 1.0) * float(height_scale)
    top = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    triangles: list[np.ndarray] = []

    # Top surface: 2 tris per cell.
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel()
    d = idx[1:, 1:].ravel()
    top_tri_1 = np.stack([top[a], top[b], top[c]], axis=1)
    top_tri_2 = np.stack([top[b], top[d], top[c]], axis=1)
    triangles.append(top_tri_1)
    triangles.append(top_tri_2)

    # Base surface (z = -base_thickness): two big triangles spanning the rect.
    z_base = -float(base_thickness)
    p00 = np.array([0.0, 0.0, z_base], dtype=np.float32)
    p10 = np.array([aspect, 0.0, z_base], dtype=np.float32)
    p11 = np.array([aspect, 1.0, z_base], dtype=np.float32)
    p01 = np.array([0.0, 1.0, z_base], dtype=np.float32)
    triangles.append(np.array([[p00, p11, p10], [p00, p01, p11]], dtype=np.float32))

    # Side walls: stitch each border row/column down to the base.
    def _wall(top_pts: np.ndarray) -> np.ndarray:
        n = top_pts.shape[0]
        out = np.empty((2 * (n - 1), 3, 3), dtype=np.float32)
        for i in range(n - 1):
            t0 = top_pts[i]
            t1 = top_pts[i + 1]
            b0 = np.array([t0[0], t0[1], z_base], dtype=np.float32)
            b1 = np.array([t1[0], t1[1], z_base], dtype=np.float32)
            out[2 * i] = np.stack([t0, t1, b0])
            out[2 * i + 1] = np.stack([t1, b1, b0])
        return out

    triangles.append(_wall(top.reshape(h, w, 3)[0, :]))     # y=0 edge
    triangles.append(_wall(top.reshape(h, w, 3)[-1, :]))    # y=1 edge
    triangles.append(_wall(top.reshape(h, w, 3)[:, 0]))     # x=0 edge
    triangles.append(_wall(top.reshape(h, w, 3)[:, -1]))    # x=aspect edge

    soup = np.concatenate(triangles, axis=0).astype(np.float32, copy=False)
    return MeshData(triangles=soup)


# ----------------------------------------------------------- writer

def write_stl_binary(mesh: MeshData, path: Path | str) -> Path:
    """Write a binary STL file (header + count + 50-byte triangle records)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tris = np.asarray(mesh.triangles, dtype=np.float32)
    n = tris.shape[0]
    with p.open("wb") as f:
        f.write(b"\x00" * _STL_HEADER_BYTES)
        f.write(struct.pack("<I", n))
        for tri in tris:
            # STL stores the face normal first; we emit zeros and let
            # downstream tools recompute. (Many slicers ignore stored normals.)
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in tri:
                f.write(struct.pack("<3f", float(v[0]), float(v[1]), float(v[2])))
            f.write(struct.pack("<H", 0))
    return p


# ----------------------------------------------------------- default backends

class _ExtrudeFromHeightmapSynth:
    """Default synthesizer: needs a precomputed heightmap (caller provides it).

    The ``image`` argument is accepted for API parity but ignored — this
    backend assumes the caller already ran the photo through the depth
    pipeline.
    """

    def infer(
        self,
        image: Optional[Image.Image] = None,
        *,
        heightmap: Optional[np.ndarray] = None,
        base_thickness: float = DEFAULT_EXTRUDE_BASE_THICKNESS,
        height_scale: float = DEFAULT_EXTRUDE_HEIGHT_SCALE,
        subsample: int = DEFAULT_EXTRUDE_SUBSAMPLE,
    ) -> MeshData:
        if heightmap is None:
            raise ValueError(
                "extrude-from-heightmap requires heightmap=... (run the "
                "depth pipeline first)."
            )
        return extrude_heightmap_to_mesh(
            heightmap,
            base_thickness=base_thickness,
            height_scale=height_scale,
            subsample=subsample,
        )


class _Hunyuan3DStub:
    """Loader-time guard for the Hunyuan3D-2-mini opt-in backend."""

    def __init__(self, device: str) -> None:
        self._device = device
        try:
            import hunyuan3d  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Hunyuan3D-2 is opt-in: pip install hunyuan3d (and accept "
                "the non-commercial license) to enable this backend."
            ) from exc

    def infer(self, image, **kwargs):
        raise RuntimeError(
            "Hunyuan3D-2 inference not wired in this build; install the "
            "package and replace _Hunyuan3DStub with the upstream pipeline."
        )


def _make_extrude_loader() -> Callable[[str], Any]:
    def _load(_device: str) -> Any:
        return _ExtrudeFromHeightmapSynth()
    return _load


# ----------------------------------------------------------- registrations

register_synthesizer(MeshSynthesisSpec(
    key="extrude-from-heightmap",
    label="Extrude heightmap (CPU, instant)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,
    loader=_make_extrude_loader(),
))


register_synthesizer(MeshSynthesisSpec(
    key="hunyuan3d-2-mini",
    label="Hunyuan3D-2 Mini (image→3D, GPU, opt-in)",
    license="Tencent NCL",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=6000,
    loader=lambda device: _Hunyuan3DStub(device),
))
