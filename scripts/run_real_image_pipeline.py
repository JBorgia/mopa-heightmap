"""Run the full zone pipeline with a real image and inspect what's generated.

Uses the user's real sculptok project from Downloads/mopa_export (3):
source_photo.png + heightmap.png (sculptok) + subject_mask.png.
Extracts the .lbrn2 bundle to <export_dir>/zone_project/ and writes a
contact sheet of every generated pass layer.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api import service_adapter

EXPORT_DIR = Path(r"c:/Users/TwentyOne21/Downloads/mopa_export (3)")
OUT_DIR = EXPORT_DIR / "zone_project"
OUT_DIR.mkdir(exist_ok=True)

client = TestClient(app)
PROFILE = "mopa_60w_brass"  # zone_params now on all material profiles

# --- 1. upload the real photo + sculptok heightmap ---------------------------
photo_bytes = (EXPORT_DIR / "source_photo.png").read_bytes()
r = client.post("/upload", files={"file": ("source_photo.png", photo_bytes, "image/png")})
r.raise_for_status()
up = r.json()
print(f"photo: {up['w']}x{up['h']} image_id={up['image_id']}")

hm_bytes = (EXPORT_DIR / "heightmap.png").read_bytes()
r = client.post("/upload/heightmap", files={"file": ("heightmap.png", hm_bytes, "image/png")})
r.raise_for_status()
hm_path = r.json()["heightmap_path"]
print(f"heightmap: {r.json()['width']}x{r.json()['height']} -> {hm_path}")

# --- 2. store the real subject mask as a blob (in-process) -------------------
mask_img = Image.open(EXPORT_DIR / "subject_mask.png").convert("L")
mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
subject_mask_id = service_adapter.blob_store.store_heightmap(mask_arr)
print(f"subject mask: {mask_arr.shape} coverage={float(mask_arr.mean())*100:.1f}% id={subject_mask_id}")

# --- 3. render with zone patterns --------------------------------------------
settings = {
    "external_heightmap_path": hm_path,
    "zone_width_mm": 50.0, "zone_height_mm": 50.0, "zone_shape": "circle",
    "zone_border_width_mm": 3.0, "zone_rim_width_mm": 1.2,
    "border_pattern": "greek_key", "border_key_count": 22,
    "rim_pattern": "reeded", "rim_reed_count": 140,
    "field_pattern": "basket_weave", "field_pattern_scale": 1.3,
    "field_pattern_depth": 0.65,
}
# mask_id = the wizard's stored subject mask — exercises the new reuse path.
r = client.post("/render", json={"image_id": up["image_id"], "settings": settings,
                                 "profile_name": PROFILE, "mask_id": subject_mask_id})
r.raise_for_status()
render = r.json()
print(f"render: heightmap_id={render['heightmap_id']} warnings={render['warnings']}")

# save the rendered (patterned) heightmap + shaded preview for inspection
for blob_id, name in [(render["heightmap_id"], "rendered_heightmap.png"),
                      (render["preview_id"], "shaded_preview.png")]:
    blob = client.get(f"/blob/{blob_id}")
    (OUT_DIR / name).write_bytes(blob.content)
    print(f"  saved {name}")

# --- 4. plan ------------------------------------------------------------------
r = client.post("/plan", json={"image_id": up["image_id"], "heightmap_id": render["heightmap_id"],
                               "profile_name": PROFILE, "settings": settings,
                               "shape_override": "circle"})
r.raise_for_status()
plan = r.json()
print(f"\nplan: {len(plan['passes'])} passes")
for p in sorted(plan["passes"], key=lambda x: x["pass_number"]):
    print(f"  fire-order {p['pass_number']}: {p['label']}")

# --- 5. export lbrn2 with the subject mask ------------------------------------
r = client.post("/export/lbrn2", json={
    "plan_id": plan["plan_id"], "heightmap_id": render["heightmap_id"],
    "profile_name": PROFILE, "subject_mask_id": subject_mask_id,
    "shape_override": "circle",
})
r.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(r.content))
zf.extractall(OUT_DIR)
print(f"\nbundle extracted to {OUT_DIR}:")
for n in zf.namelist():
    print(f"  {n} ({len(zf.read(n)):,} bytes)")

# --- 6. layer table from the XML ----------------------------------------------
xml = (OUT_DIR / "project.lbrn2").read_text(encoding="utf-8")
print("\nLightBurn layers (CutSetting_Img):")
for m in re.finditer(r"<CutSetting_Img[^>]*>(.*?)</CutSetting_Img>", xml, re.S):
    body = m.group(1)
    def g(tag: str) -> str:
        mm = re.search(rf'<{tag} Value="([^"]*)"', body)
        return mm.group(1) if mm else "-"
    print(f"  [{g('index')}] {g('name'):<12} speed={g('speed'):>5} power={g('maxPower'):>3}% "
          f"freq={int(g('frequency'))//1000}kHz pulse={g('QPulseWidth')}ns passes={g('numPasses')}")

# --- 7. contact sheet of all generated pieces ----------------------------------
tiles = []
for f in sorted(OUT_DIR.glob("*.png")):
    if f.name == "layers_contact_sheet.png":
        continue
    img = Image.open(f).convert("L").resize((400, 400), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 399, 18], fill=(20, 20, 20))
    d.text((4, 3), f.name, fill=(255, 220, 80))
    tiles.append(img)
cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 404, rows * 404), (40, 40, 40))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 404 + 2, (i // cols) * 404 + 2))
sheet_path = OUT_DIR / "layers_contact_sheet.png"
sheet.save(sheet_path)
print(f"\ncontact sheet: {sheet_path}")
