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
from ..service_adapter import do_export_lbrn2, do_export_png, get_profile_data
from .. import blob_store

router = APIRouter(prefix="/export", tags=["export"])


def _append_quad(
    triangles: list[list[list[float]]],
    p0: list[float],
    p1: list[float],
    p2: list[float],
    p3: list[float],
) -> None:
    """Append two STL triangles that form a quad."""
    triangles.append([p0, p1, p2])
    triangles.append([p0, p2, p3])


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
    z_base = -float(base_thickness_mm)

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

    if base_thickness_mm > 0.0 and h >= 2 and w >= 2:
        max_x = float(w - 1)
        max_y = float(h - 1)

        # Bottom face.
        _append_quad(
            verts,
            [0.0, 0.0, z_base],
            [0.0, max_y, z_base],
            [max_x, max_y, z_base],
            [max_x, 0.0, z_base],
        )

        # Front and back walls.
        for c in range(w - 1):
            x0, x1 = float(c), float(c + 1)
            _append_quad(
                verts,
                [x0, 0.0, z_base],
                [x1, 0.0, z_base],
                [x1, 0.0, float(z[0, c + 1])],
                [x0, 0.0, float(z[0, c])],
            )
            _append_quad(
                verts,
                [x0, max_y, z_base],
                [x0, max_y, float(z[h - 1, c])],
                [x1, max_y, float(z[h - 1, c + 1])],
                [x1, max_y, z_base],
            )

        # Left and right walls.
        for r in range(h - 1):
            y0, y1 = float(r), float(r + 1)
            _append_quad(
                verts,
                [0.0, y0, z_base],
                [0.0, y0, float(z[r, 0])],
                [0.0, y1, float(z[r + 1, 0])],
                [0.0, y1, z_base],
            )
            _append_quad(
                verts,
                [max_x, y0, z_base],
                [max_x, y1, z_base],
                [max_x, y1, float(z[r + 1, w - 1])],
                [max_x, y0, float(z[r, w - 1])],
            )

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
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    """Bundle the user-selected formats + every available reference
    artifact into a single zip.

    Drives the wizard's Submit action. The ``include_*`` flags gate the
    heavy outputs (PNG / .lbrn2 / STL). Optional reference artifacts
    (subject mask, source photo, sculptok input, profile YAML) are
    ALWAYS included when their blob ids are supplied — losing them in
    the export forces the user to re-run the wizard, which is worse
    than a slightly bigger zip.
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
                    # Forward the mask so it lands as a non-engraving
                    # toggleable layer in the .lbrn2 itself, not just a
                    # standalone PNG in the zip. Users get both — the
                    # layer for in-LightBurn workflows, the PNG for
                    # external tooling.
                    subject_mask_id=req.subject_mask_id,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
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

        # ----------------- always-on reference artifacts ------------------
        # Subject mask — when the user computed one (via /mask, /render, or
        # /sculptok/generate) they almost certainly want it shipped to
        # LightBurn. The mask drives engrave/no-engrave at run time.
        if req.subject_mask_id:
            mask_bytes = blob_store.load_bytes(req.subject_mask_id)
            if mask_bytes is not None:
                zf.writestr("subject_mask.png", mask_bytes)

        # Original photo — useful as a sanity-check reference when the
        # heightmap looks off and the user wants to compare.
        if req.image_id:
            src_bytes = blob_store.load_bytes(req.image_id)
            if src_bytes is not None:
                zf.writestr("source_photo.png", src_bytes)

        # The actual photo Sculptok generated the depth map from. Lets the
        # user see what the model saw (post-prep + bg-replace) without
        # trusting that the wizard rendered it identically.
        if req.sculptok_input_id:
            sin_bytes = blob_store.load_bytes(req.sculptok_input_id)
            if sin_bytes is not None:
                zf.writestr("sculptok_input.png", sin_bytes)

        # Profile YAML — so users can switch profiles in LightBurn without
        # re-exporting. Includes machine, lightburn_mode, every cut-setting
        # field. Stays valid even when the user picks a different material
        # cut later (they just disable the brass layer in LightBurn and
        # add their own).
        if req.profile_name:
            try:
                profile_data = get_profile_data(req.profile_name)
                # Persist as YAML for round-trip with the CLI / mopa.profiles
                # loader. Falls back to JSON if PyYAML isn't available.
                try:
                    import yaml
                    profile_text = yaml.safe_dump(profile_data, sort_keys=False)
                except ImportError:
                    import json
                    profile_text = json.dumps(profile_data, indent=2)
                zf.writestr(f"profile_{req.profile_name}.yaml", profile_text)
            except Exception:
                # Profile lookup is best-effort — never fail the whole bundle
                # because of a missing/unreadable profile file.
                pass

    headers = {"Content-Disposition": "attachment; filename=mopa_export.zip"}
    if skipped:
        # Surface skipped formats in a custom header so the client can toast.
        headers["X-Mopa-Skipped"] = ",".join(skipped)
    return Response(content=out.getvalue(), media_type="application/zip", headers=headers)
