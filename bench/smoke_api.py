"""End-to-end API smoke test: upload → render → plan → export/lbrn2 zip."""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import apps.api.service_adapter as adapter
from zoedepth.laser.service import HeightmapService
from zoedepth.laser.settings import AppSettings


class _FakeBumpModel:
    def infer_pil(self, image, pad_input=True, with_flip_aug=True):
        w, h = image.size
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        cy, cx = h / 2.0, w / 2.0
        bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (max(w, h) * 0.18) ** 2))
        return (1.0 - 0.7 * bump).astype(np.float32)


def main() -> None:
    adapter._svc = HeightmapService(
        app_settings=AppSettings(),
        model_loader=lambda name, device: (_FakeBumpModel(), device),
    )
    from apps.api.main import app
    client = TestClient(app, raise_server_exceptions=True)

    buf = io.BytesIO()
    arr = np.full((128, 128, 3), 140, dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    r = client.post("/upload", files={"file": ("king.png", buf.getvalue(), "image/png")})
    r.raise_for_status()
    image_id = r.json()["image_id"]
    print(f"1. /upload         OK  image_id={image_id[:12]}...")

    r = client.post("/render", json={"image_id": image_id})
    r.raise_for_status()
    heightmap_id = r.json()["heightmap_id"]
    print(f"2. /render         OK  heightmap_id={heightmap_id[:12]}...  "
          f"elapsed={r.json()['elapsed_s']:.2f}s")

    r = client.post("/plan", json={
        "image_id": image_id,
        "heightmap_id": heightmap_id,
        "profile_name": "Colour60W-M7",
    })
    r.raise_for_status()
    plan_id = r.json()["plan_id"]
    passes = r.json()["passes"]
    print(f"3. /plan           OK  plan_id={plan_id[:12]}...  passes={len(passes)}")
    for p in passes:
        print(f"                       pass_{p['pass_number']}: {p['label']}")

    r = client.post("/export/lbrn2", json={"plan_id": plan_id, "heightmap_id": heightmap_id})
    r.raise_for_status()
    print(f"4. /export/lbrn2   OK  ({len(r.content)} bytes, "
          f"content-type={r.headers['content-type']})")

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        print("5. zip contents:")
        for n in names:
            info = zf.getinfo(n)
            print(f"       {n} ({info.file_size} bytes)")
        assert "project.lbrn2" in names
        pngs = [n for n in names if n.endswith(".png")]
        assert len(pngs) == len(passes), f"expected {len(passes)} pngs, got {len(pngs)}"
        project_bytes = zf.read("project.lbrn2")

    root = ET.fromstring(project_bytes)
    assert root.tag == "LightBurnProject"
    shapes = root.findall("Shape")
    cut_settings = root.findall("CutSetting")
    print(f"6. project parse   OK  cut_settings={len(cut_settings)} shapes={len(shapes)}")

    # Verify every shape's SourceFile resolves to a real PNG inside the zip.
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zip_names = set(zf.namelist())
        for s in shapes:
            src = s.attrib.get("SourceFile")
            assert src in zip_names, f"shape references missing PNG: {src}"
    print("7. all shape SourceFile attrs resolve inside the zip  OK")

    print()
    print("ALL ENDPOINTS PASSED")


if __name__ == "__main__":
    main()
