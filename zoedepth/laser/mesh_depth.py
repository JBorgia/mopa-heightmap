"""Mesh-based depth: image → 3D mesh (TripoSR) → orthographic Z → heightmap.

Why this exists: monocular depth networks (DAv2, ZoeDepth, Marigold) and
even diffusion-stylized variants of those produce depth maps that are
"photographically plausible" but lack *sculptural confidence*. They know
where the body is, but they don't know that fingers are separate, that
fur has individual strands, or that a face has cheekbone planes — those
are topological facts a depth network has to infer, and it usually
won't.

A 3D mesh prior solves this. Image-to-3D models like TripoSR generate a
genuine 3D mesh from a single photo; rendering the front-facing
Z-buffer of that mesh gives us a heightmap with the topological
correctness baked in. This is the architectural difference between our
pipeline (depth network ⇒ smooth heightmap) and reference targets like
sculptok (mesh ⇒ orthographic depth ⇒ sculptural heightmap).

Default backend: **TripoSR** (Stability AI, 2024). MIT licence, ~540 M
params, ~2.4 GB weights, runs on CPU in ~30-60 s and on a 4 GB GPU in
~10-15 s. Generates a textured mesh; we discard the texture and
orthographically render the front-facing Z to a depth array.

**Setup before use** — the vendored TripoSR architecture has been
removed from the repo to keep it light. To re-enable this backend:

    cd <repo>
    git clone --depth=1 https://github.com/VAST-AI-Research/TripoSR.git vendor/triposr
    # then edit vendor/triposr/tsr/models/isosurface.py to swap
    # ``from torchmcubes import marching_cubes`` for the PyMCubes shim
    # documented at the top of that file (or just `pip install pymcubes`
    # and use the in-process monkey-patch we ship in this module).

Why removed: the experiment showed TripoSR produces a too-blobby mesh
for complex portrait poses (raised arms / V-pose) at 4 GB VRAM, so it
isn't on the recommended path. See
``memory/feedback_sculptok_parity_ceiling.md`` for the verdict.

The Python wrapper here stays as opt-in scaffolding so revisiting
mesh-based depth (with Hunyuan3D, TRELLIS, or a better TripoSR variant)
in the future is a swap-the-loader change rather than starting over.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "TripoSRDepthBackend",
    "load_triposr_depth_backend",
    "DEFAULT_MC_RESOLUTION",
    "DEFAULT_FOREGROUND_RATIO",
    "ORTHO_GRID_PIXELS",
]


# Marching-cubes grid resolution for mesh extraction. 256 is the upstream
# default; lower (128) trades surface fidelity for ~3× speed.
DEFAULT_MC_RESOLUTION: int = 256

# What fraction of the frame the foreground subject should occupy after
# automatic background removal. Matches TripoSR's published defaults.
DEFAULT_FOREGROUND_RATIO: float = 0.85

# Number of pixels along each axis when ortho-rendering the mesh's
# Z-buffer. Output gets resampled to the input image's native size by
# the caller; this controls the precision of the ray-cast.
ORTHO_GRID_PIXELS: int = 1024


def _ensure_vendor_on_path() -> None:
    """Add ``vendor/triposr`` to ``sys.path`` so its ``tsr`` package imports.

    Raises a clear, actionable error if the directory is missing — that's
    the most common case (the vendor tree is not committed; see this
    module's docstring for the re-clone instructions).
    """
    vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "triposr"
    if not vendor_root.exists():
        raise RuntimeError(
            "TripoSR vendor tree not present at "
            f"{vendor_root}. Re-clone with:\n"
            "    git clone --depth=1 "
            "https://github.com/VAST-AI-Research/TripoSR.git vendor/triposr\n"
            "and apply the PyMCubes shim at vendor/triposr/tsr/models/isosurface.py "
            "(see this module's docstring)."
        )
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))


@dataclass(frozen=True)
class _MeshOrthoResult:
    """One render's worth of depth + diagnostics."""

    depth: np.ndarray            # (H, W) float32, "larger = farther from camera"
    mesh_vertex_count: int
    mesh_triangle_count: int


class TripoSRDepthBackend:
    """Adapter giving the TripoSR mesh pipeline a ZoeDepth-style ``infer_pil``.

    The pipeline:
        1. Background-remove the input (rembg) and center the subject at
           ``DEFAULT_FOREGROUND_RATIO`` of a square gray canvas.
        2. Run TripoSR forward → triplane scene codes.
        3. Marching-cubes extract a mesh at ``mc_resolution``.
        4. Orthographically ray-cast the mesh from a known camera angle
           into an ``ORTHO_GRID_PIXELS²`` Z-buffer.
        5. Resample to the input image's native resolution.
        6. Convert "z (closer = larger)" to ZoeDepth's "larger = farther"
           convention so downstream stages don't notice the swap.
    """

    def __init__(self, model: Any, device: str, mc_resolution: int = DEFAULT_MC_RESOLUTION) -> None:
        self._model = model
        self._device = device
        self._mc_resolution = int(mc_resolution)
        # rembg session is built lazily on first call.
        self._rembg_session: Any | None = None

    # ------------------------------------------------------------------ public

    def infer_pil(
        self,
        image: Image.Image,
        pad_input: bool = True,
        with_flip_aug: bool = True,
    ) -> np.ndarray:
        """Return a 2-D float32 depth array at the input image's resolution.

        The ``pad_input`` / ``with_flip_aug`` kwargs exist for API parity
        with ZoeDepth's ``infer_pil`` and are ignored — TripoSR has its
        own input pre-processing pipeline.
        """
        del pad_input  # accepted for API parity, unused
        del with_flip_aug

        target_w, target_h = image.size
        prepped = self._preprocess(image)
        scene_codes = self._run_model(prepped)
        mesh = self._extract_mesh(scene_codes)
        depth = self._ortho_render_depth(mesh, ORTHO_GRID_PIXELS)
        return self._resample_and_orient(depth, target_w, target_h)

    # ----------------------------------------------------------------- internals

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Background-remove + center on gray, matching TripoSR's run.py."""
        from tsr.utils import remove_background, resize_foreground

        if self._rembg_session is None:
            import rembg
            self._rembg_session = rembg.new_session()
        rgba = remove_background(image.convert("RGB"), self._rembg_session)
        rgba = resize_foreground(rgba, DEFAULT_FOREGROUND_RATIO)
        arr = np.asarray(rgba).astype(np.float32) / 255.0
        # Composite on mid-gray (matches TripoSR's training distribution).
        comp = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((comp * 255.0).astype(np.uint8))

    def _run_model(self, prepped: Image.Image):
        import torch
        with torch.no_grad():
            return self._model([prepped], device=self._device)

    def _extract_mesh(self, scene_codes):
        # ``has_vertex_color=True`` keeps a colour we don't use, but is
        # the supported call signature; the colours come back attached
        # to the trimesh and just sit there.
        meshes = self._model.extract_mesh(
            scene_codes, has_vertex_color=True, resolution=self._mc_resolution,
        )
        return meshes[0]

    def _ortho_render_depth(self, mesh, grid: int) -> np.ndarray:
        """Cast a regular grid of rays in -Z and record first-hit Z.

        TripoSR's mesh sits in ``[0, 1]³`` (after the marching-cubes
        rescale in :class:`MarchingCubeHelper`). After centering, the
        mesh extents fall inside roughly ``[-0.5, 0.5]³``. We sample
        the XY plane on a uniform grid, ray-cast in -Z, and take the
        first-intersection Z as the surface height. Pixels with no hit
        get the background plane (``z = -∞`` → mapped to "far").
        """
        import trimesh

        # Center + normalise the mesh to the unit cube so the projection
        # math doesn't depend on TripoSR's internal coordinate frame.
        bbox = mesh.bounds                     # (2, 3)
        size = float(np.max(bbox[1] - bbox[0]))
        centered = mesh.copy()
        centered.apply_translation(-(bbox[0] + bbox[1]) * 0.5)
        if size > 0:
            centered.apply_scale(1.0 / size)

        # The TripoSR mesh's *front-facing* axis depends on its training
        # convention. Empirically, +Z points toward the camera in the
        # rendered NeRF views; we cast rays in -Z from above the mesh
        # and grab the first hit.
        xs = np.linspace(-0.5, 0.5, grid, dtype=np.float64)
        ys = np.linspace(0.5, -0.5, grid, dtype=np.float64)  # PIL Y is top-down
        xx, yy = np.meshgrid(xs, ys)
        origins = np.stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)],
            axis=-1,
        )
        directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))

        # ``intersects_first`` returns the index of the first triangle
        # each ray hits, plus the location. We want the Z of the
        # location.
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(centered)
        locations, ray_idx, _ = intersector.intersects_location(
            origins, directions, multiple_hits=False,
        )
        depth = np.full(grid * grid, -10.0, dtype=np.float32)  # bg = far
        if locations.size > 0:
            depth[ray_idx] = locations[:, 2].astype(np.float32)
        return depth.reshape(grid, grid)

    def _resample_and_orient(
        self, depth_grid: np.ndarray, target_w: int, target_h: int,
    ) -> np.ndarray:
        """Resample to the input image's resolution and flip to ZoeDepth polarity.

        TripoSR's depth at this point is "larger Z = closer to camera".
        ZoeDepth's convention (and the rest of our pipeline) is
        "larger value = farther from camera". We negate so the rest of
        the pipeline doesn't notice.
        """
        # Resample with PIL bilinear (target aspect may differ).
        as_pil = Image.fromarray(depth_grid, mode="F").resize(
            (target_w, target_h), Image.BILINEAR,
        )
        depth = np.asarray(as_pil, dtype=np.float32)
        # Flip polarity: closer→larger becomes closer→smaller.
        depth = -depth
        return depth


def load_triposr_depth_backend(
    device: str,
    *,
    mc_resolution: int = DEFAULT_MC_RESOLUTION,
    repo: str = "stabilityai/TripoSR",
) -> TripoSRDepthBackend:
    """Lazy-load TripoSR weights + return a depth backend.

    Vendor patch path: this requires ``vendor/triposr/`` to be present
    (cloned from the upstream GitHub) and the PyMCubes monkey-patch in
    ``vendor/triposr/tsr/models/isosurface.py`` to be applied. Both are
    in-tree.
    """
    _ensure_vendor_on_path()
    from tsr.system import TSR
    import torch

    model = TSR.from_pretrained(
        repo, config_name="config.yaml", weight_name="model.ckpt",
    )
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    target_device = device if is_cuda else "cpu"
    model.renderer.set_chunk_size(2048 if is_cuda else 4096)
    model.to(target_device)
    return TripoSRDepthBackend(
        model, device=target_device, mc_resolution=mc_resolution,
    )
