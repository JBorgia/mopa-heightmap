"""Replay the mopa_export (12) scenario with the fixes in place.

Uses the exact artifacts the user exported: the crest photo and the
sculptok heightmap WITHOUT any baked patterns. Verifies that the
exported .lbrn2 zone layers now carry real pattern relief, the device
mask comes from the heightmap (not the blobby photo mask), and layers
have readable names.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from apps.api.main import app

SRC = Path(r"c:/Users/TwentyOne21/Downloads/mopa_export (12)")
OUT = SRC / "fixed_export"
OUT.mkdir(exist_ok=True)
client = TestClient(app)
PROFILE = "mopa_60w_brass"

image_id = client.post("/upload", files={"file": ("p.png", (SRC / "source_photo.png").read_bytes(), "image/png")}).json()["image_id"]
hm_path = client.post("/upload/heightmap", files={"file": ("h.png", (SRC / "heightmap.png").read_bytes(), "image/png")}).json()["heightmap_path"]

settings = {
    "external_heightmap_path": hm_path,
    "heightmap_enhance_mode": "off",   # keep their heightmap exactly as-is
    "zone_width_mm": 60.0, "zone_height_mm": 60.0, "zone_shape": "circle",
    "zone_border_width_mm": 4.0, "zone_rim_width_mm": 1.5,
    "border_pattern": "greek_key", "border_key_count": 24,
    "rim_pattern": "denticled", "rim_element_count": 110,
    "field_pattern": "guilloche", "field_pattern_depth": 0.6,
    "pre_clean_enabled": True,
    "photo_tonal_enabled": True,
}
r = client.post("/render", json={"image_id": image_id, "settings": settings, "profile_name": PROFILE})
r.raise_for_status()
hm_id = r.json()["heightmap_id"]
clean_id = r.json().get("clean_heightmap_id")
print(f"render: warnings={r.json()['warnings']} clean_heightmap_id={'set' if clean_id else 'MISSING'}")

plan = client.post("/plan", json={"image_id": image_id, "heightmap_id": hm_id,
                                  "profile_name": PROFILE, "settings": settings,
                                  "shape_override": "circle"}).json()
print(f"plan: {len(plan['passes'])} passes")

# Export WITHOUT any subject_mask_id: the device mask must come from the
# clean heightmap itself now.
r = client.post("/export/lbrn2", json={"plan_id": plan["plan_id"], "heightmap_id": hm_id,
                                       "profile_name": PROFILE, "shape_override": "circle",
                                       "clean_heightmap_id": clean_id})
r.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(r.content))
zf.extractall(OUT)
print("bundle:", zf.namelist())

xml = (OUT / "project.lbrn2").read_text(encoding="utf-8")
print("\nlayers:")
for m in re.finditer(r"<CutSetting_Img[^>]*>(.*?)</CutSetting_Img>", xml, re.S):
    body = m.group(1)
    def g(t, b=body):
        mm = re.search(rf'<{t} Value="([^"]*)"', b)
        return mm.group(1) if mm else "-"
    print(f"  [{g('index')}] {g('name'):<12} speed={g('speed'):>5} power={g('maxPower'):>3}% numPasses={g('numPasses')}")

# Contact sheet of the new pass PNGs.
tiles = []
for f in sorted(OUT.glob("pass_*.png")):
    img = Image.open(f).convert("L").resize((360, 360)).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 359, 16], fill=(15, 15, 15))
    d.text((4, 2), f.name, fill=(255, 220, 80))
    tiles.append(img)
cols = 4
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 364, rows * 364), (40, 40, 40))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 364 + 2, (i // cols) * 364 + 2))
sheet.save(OUT / "fixed_passes_sheet.png")
print("\nsheet:", OUT / "fixed_passes_sheet.png")

# Quantitative checks: border layer must contain pattern relief (many
# intermediate grey levels inside the ring), not just a flat mask.
for name, expect in [("zone_border", "greek key"), ("zone_field", "guilloche"), ("zone_rim", "denticles")]:
    f = next(OUT.glob(f"pass_*_{name}.png"))
    a = np.asarray(Image.open(f).convert("L"), dtype=np.int32)
    engrave = a < 250
    interior = a[engrave]
    uniq = len(np.unique(interior))
    print(f"{f.name}: engrave_px={engrave.sum():,} grey_levels_in_zone={uniq} ({expect})")
