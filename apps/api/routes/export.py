"""POST /export/png, /export/lbrn2, /export/stl, /export/bundle — export finished artifacts."""
from __future__ import annotations

import io
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..schemas import (
    ExportBundleRequest,
    ExportLbrn2Request,
    ExportPngRequest,
    ExportStlRequest,
)
from ..service_adapter import do_export_lbrn2, do_export_png
from .. import blob_store

router = APIRouter(prefix="/export", tags=["export"])


def _build_stl_bytes(heightmap_id: str, z_scale_mm: float, base_thickness_mm: float) -> bytes:
    """Render a heightmap blob into binary STL bytes.

    Shared by ``POST /export/stl`` and the ``/export/bundle`` endpoint so a
    future numpy-stl bump only needs to be addressed in one place. Raises
    ``KeyError`` when the blob is missing and ``ImportError`` when the
    optional ``numpy-stl`` dependency is not installed — the route layer
    maps both to appropriate HTTP status codes.
    """
    hm = blob_store.load_heightmap(heightmap_id)
    if hm is None:
        raise KeyError(f"Unknown heightmap_id: {heightmap_id!r}")

    import numpy as np
    # numpy-stl moved Mode to the top-level `stl` module; the older
    # ``stl.mesh.Mode`` path raises AttributeError in current versions.
    from stl import mesh as stl_mesh, Mode as STLMode  # type: ignore[import]

    h, w = hm.shape
    z = hm * z_scale_mm
    _ = -base_thickness_mm  # base extrusion not yet wired through; reserved.

    # Build quad mesh as two triangles per cell.
    verts: list[list[list[float]]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            x0, x1 = float(c), float(c + 1)
            y0, y1 = float(r), float(r + 1)
            z00 = float(z[r, c])
            z10 = float(z[r, c + 1])
            z01 = float(z[r + 1, c])
            z11 = float(z[r + 1, c + 1])
            verts.append([[x0, y0, z00], [x1, y0, z10], [x0, y1, z01]])
            verts.append([[x1, y0, z10], [x1, y1, z11], [x0, y1, z01]])

    arr = np.array(verts, dtype=np.float32)
    m = stl_mesh.Mesh(np.zeros(len(arr), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(arr):
        m.vectors[i] = tri
    buf = io.BytesIO()
    m.save("out.stl", fh=buf, mode=STLMode.BINARY)
    return buf.getvalue()


@router.post("/png")
async def export_png(req: ExportPngRequest) -> Response:
    try:
        data = do_export_png(req.heightmap_id, req.bit_depth)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=heightmap.png"},
    )


@router.post("/lbrn2")
async def export_lbrn2(req: ExportLbrn2Request) -> Response:
    """Emit a LightBurn .lbrn2 bundle (project + per-pass PNGs) as a zip.

    The .lbrn2 file alone references the per-pass PNGs by relative path;
    distributing them as a single zip keeps the bundle self-contained so
    LightBurn finds every bitmap when the user unpacks and opens the
    project.
    """
    try:
        data = do_export_lbrn2(
            plan_id=req.plan_id,
            heightmap_id=req.heightmap_id,
            profile_name=req.profile_name,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=project.lbrn2.zip"},
    )


@router.post("/stl")
async def export_stl(req: ExportStlRequest) -> Response:
    """Convert a heightmap blob to an STL binary.  Requires numpy-stl."""
    try:
        data = _build_stl_bytes(req.heightmap_id, req.z_scale_mm, req.base_thickness_mm)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"STL export requires numpy-stl: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return Response(
        content=data,
        media_type="model/stl",
        headers={"Content-Disposition": "attachment; filename=heightmap.stl"},
    )


@router.post("/bundle")
async def export_bundle(req: ExportBundleRequest) -> Response:
    """Bundle the user-selected formats into a single zip.

    Drives the wizard's Submit action: rather than firing three separate
    downloads, the user picks the formats they want and gets one
    ``mopa_export.zip`` they can drop into a directory of their choice.

    The bundle inlines the .lbrn2 contents (project file + per-pass PNGs)
    at the top level of the zip, so opening the zip in LightBurn just
    works. The .lbrn2 export endpoint already returns a self-contained
    zip, so we extract its members here rather than nesting zip-in-zip.
    """
    if not (req.include_png or req.include_lbrn2 or req.include_stl):
        raise HTTPException(
            status_code=422,
            detail="Bundle requires at least one of include_png / include_lbrn2 / include_stl.",
        )
    if req.include_lbrn2 and not req.plan_id:
        raise HTTPException(
            status_code=422,
            detail="include_lbrn2=true requires plan_id; compute the pass plan first.",
        )

    out = io.BytesIO()
    skipped: list[str] = []
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if req.include_png:
            try:
                zf.writestr("heightmap.png", do_export_png(req.heightmap_id, 16))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        if req.include_lbrn2:
            try:
                inner = do_export_lbrn2(
                    plan_id=req.plan_id,  # type: ignore[arg-type]  # validated above
                    heightmap_id=req.heightmap_id,
                    profile_name=req.profile_name,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            # do_export_lbrn2 returns its own zip — inline its members so the
            # outer zip is flat and LightBurn opens project.lbrn2 directly.
            with zipfile.ZipFile(io.BytesIO(inner), mode="r") as inner_zf:
                for name in inner_zf.namelist():
                    zf.writestr(name, inner_zf.read(name))

        if req.include_stl:
            try:
                zf.writestr(
                    "heightmap.stl",
                    _build_stl_bytes(req.heightmap_id, req.z_scale_mm, req.base_thickness_mm),
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ImportError:
                # STL is the most fragile (optional native dep). Skip rather
                # than fail the whole bundle so the user still gets PNG/.lbrn2.
                skipped.append("stl")

    headers = {"Content-Disposition": "attachment; filename=mopa_export.zip"}
    if skipped:
        # Surface skipped formats in a custom header so the client can toast.
        headers["X-Mopa-Skipped"] = ",".join(skipped)
    return Response(content=out.getvalue(), media_type="application/zip", headers=headers)
