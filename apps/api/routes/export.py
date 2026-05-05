"""POST /export/png, /export/lbrn2, /export/stl — export finished artifacts."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..schemas import ExportLbrn2Request, ExportPngRequest, ExportStlRequest
from ..service_adapter import do_export_lbrn2, do_export_png
from .. import blob_store

router = APIRouter(prefix="/export", tags=["export"])


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
    hm = blob_store.load_heightmap(req.heightmap_id)
    if hm is None:
        raise HTTPException(status_code=404, detail=f"Unknown heightmap_id: {req.heightmap_id!r}")
    try:
        import numpy as np
        from stl import mesh as stl_mesh  # type: ignore[import]

        h, w = hm.shape
        z = hm * req.z_scale_mm
        base = -req.base_thickness_mm

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
        import io as _io
        buf = _io.BytesIO()
        m.save("out.stl", fh=buf, mode=stl_mesh.Mode.BINARY)
        return Response(
            content=buf.getvalue(),
            media_type="model/stl",
            headers={"Content-Disposition": "attachment; filename=heightmap.stl"},
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"STL export requires numpy-stl: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
