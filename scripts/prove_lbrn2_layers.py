"""End-to-end proof: zoned render -> plan -> /export/lbrn2 -> layered project file."""
import io
import json
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)

# --- 1. synthetic photo + heightmap -----------------------------------------
rng = np.random.default_rng(7)
photo = (rng.uniform(60, 200, (300, 300, 3))).astype(np.uint8)
buf = io.BytesIO(); Image.fromarray(photo).save(buf, format="PNG")
r = client.post("/upload", files={"file": ("photo.png", buf.getvalue(), "image/png")})
r.raise_for_status(); image_id = r.json()["image_id"]

yy, xx = np.mgrid[:300, :300].astype(np.float32)
d = np.sqrt((xx - 150) ** 2 + (yy - 150) ** 2)
hm = np.clip(1.0 - d / 150.0, 0, 1)  # dome
buf = io.BytesIO(); Image.fromarray((hm * 65535).astype(np.uint16)).save(buf, format="PNG")
r = client.post("/upload/heightmap", files={"file": ("hm.png", buf.getvalue(), "image/png")})
r.raise_for_status(); hm_path = r.json()["heightmap_path"]

profile_name = "mopa_60w_sterling_silver"  # only profile with zone_params
print(f"profile: {profile_name}")

settings = {
    "external_heightmap_path": hm_path,
    "zone_width_mm": 30.0, "zone_height_mm": 30.0, "zone_shape": "circle",
    "zone_border_width_mm": 3.0, "zone_rim_width_mm": 1.0,
    "border_pattern": "rope_twist", "rim_pattern": "beaded",
    "field_pattern": "guilloche",
}

# --- 2. render ----------------------------------------------------------------
r = client.post("/render", json={"image_id": image_id, "settings": settings, "profile_name": profile_name})
r.raise_for_status(); render = r.json()
print(f"render ok, warnings={render['warnings']}")

# --- 3. plan ------------------------------------------------------------------
r = client.post("/plan", json={"image_id": image_id, "heightmap_id": render["heightmap_id"],
                               "profile_name": profile_name, "settings": settings,
                               "shape_override": "circle"})
r.raise_for_status(); plan = r.json()
print(f"plan {plan['plan_id']}: {len(plan['passes'])} passes")
for p in plan["passes"]:
    print(f"  pass {p['pass_number']}: {p['label']} depth={p['depth_um']}um color={p['color_hex']}")

# --- 4. export lbrn2 ------------------------------------------------------------
r = client.post("/export/lbrn2", json={"plan_id": plan["plan_id"], "heightmap_id": render["heightmap_id"],
                                       "profile_name": profile_name, "shape_override": "circle"})
r.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(r.content))
print(f"\nzip contents: {zf.namelist()}")

xml = zf.read("project.lbrn2").decode("utf-8")
root = ET.fromstring(xml)
print("\nCutSetting layers in project.lbrn2:")
for cs in root.iter("CutSetting"):
    idx = cs.findtext("index") or (cs.find("index").get("Value") if cs.find("index") is not None else "?")
    name = cs.findtext("name") or (cs.find("name").get("Value") if cs.find("name") is not None else "?")
    np_el = cs.find("numPasses")
    npv = np_el.get("Value") if np_el is not None else "?"
    print(f"  layer index={idx} name={name!r} numPasses={npv}")

print("\nShapes:")
for sh in root.iter("Shape"):
    t = sh.get("Type"); ci = sh.get("CutIndex")
    f = sh.get("File") or sh.get("SourceFile") or ""
    print(f"  Shape type={t} cutIndex={ci} file={f} maskId={sh.get('MaskID')}")

# --- 5. verify per-pass PNGs actually differ (zone masking applied) -------------
pngs = [n for n in zf.namelist() if n.startswith("pass_")]
arrs = {n: np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("L"), dtype=np.int32) for n in pngs}
names = sorted(arrs)
print("\nPer-pass PNG diff (nonzero-engrave px, i.e. value<255):")
for n in names:
    a = arrs[n]
    print(f"  {n}: shape={a.shape} engrave_px={(a < 250).sum()}")
for i in range(len(names) - 1):
    diff = int((arrs[names[i]] != arrs[names[i + 1]]).sum())
    print(f"  {names[i]} vs {names[i+1]}: {diff} differing px")
print("\nPROOF COMPLETE")
