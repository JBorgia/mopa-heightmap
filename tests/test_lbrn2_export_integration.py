"""End-to-end .lbrn2 export integration test.

Exercises the full path the CLI / API rely on:

    HeightmapService.render -> plan_passes -> write_lbrn ->
    load_lightburn_card -> assert ColorEntry round-trips + per-pass PNG
    paths exist on disk + the project XML is well-formed LightBurn
    output (passes the same parser the importer uses for the canonical
    color cards).

This is the "is the export pipeline actually shipping a usable file"
test. If it ever fails, the user can't drag the bundle into LightBurn.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mopa.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    load_lightburn_card,
)
from mopa.service import (
    ExportRequest,
    HeightmapService,
    merge_profile_settings,
)
from mopa.settings import AppSettings


def _write_synthetic_heightmap(target: Path, w: int = 96, h: int = 96) -> Path:
    """Write a small Gaussian-bump heightmap PNG to ``target``.

    bright_raised polarity (sculptok convention): bright center = raised.
    """
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (max(w, h) * 0.18) ** 2))
    arr = (0.3 + 0.7 * bump).astype(np.float32)
    arr16 = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr16, mode="I;16").save(target)
    return target


@pytest.fixture
def service():
    return HeightmapService(app_settings=AppSettings())


@pytest.fixture
def synthetic_image():
    arr = np.full((96, 96, 3), 140, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def settings_with_heightmap(tmp_path):
    """Settings dict that points at a synthetic heightmap PNG fixture."""
    heightmap_path = _write_synthetic_heightmap(tmp_path / "fixture_heightmap.png")
    settings = merge_profile_settings(
        None,
        {"external_heightmap_path": str(heightmap_path)},
    )
    return settings


# ----------------------------------------------------------- service.export()

def test_export_lbrn2_writes_project_and_pngs(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    request = ExportRequest(
        output_dir=tmp_path,
        base_stem="bundle",
        write_preview=False,
        write_lbrn2=True,
        write_pass_pngs=True,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)

    assert bundle.lbrn2_path is not None
    assert bundle.lbrn2_path.exists()
    tree = ET.parse(bundle.lbrn2_path)
    root = tree.getroot()
    assert root.tag == "LightBurnProject"

    assert bundle.pass_png_paths
    final_dir = tmp_path / "bundle" / "final"
    for png_path in bundle.pass_png_paths.values():
        assert png_path.exists()
        assert png_path.parent == final_dir
        assert png_path.parent == bundle.lbrn2_path.parent
        assert png_path.suffix == ".png"


def test_export_lbrn2_round_trips_through_card_loader(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """Exported project must parse back to a MaterialProfile whose entries
    match the source card's bit-for-bit (the round-trip contract from
    ``lbrn_writer.py``)."""
    request = ExportRequest(
        output_dir=tmp_path,
        base_stem="roundtrip",
        write_preview=False,
        write_lbrn2=True,
        write_pass_pngs=True,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)

    source = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    written = load_lightburn_card(bundle.lbrn2_path)

    for entry in written.entries:
        original = source.by_index.get(entry.index)
        assert original is not None, f"Wrote unknown index {entry.index}"
        for key, value in original.raw.items():
            assert entry.raw.get(key) == value, (
                f"Field {key!r} drifted on round-trip "
                f"({value!r} -> {entry.raw.get(key)!r})"
            )


def test_export_lbrn2_embeds_bitmap_data(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """LightBurn renders bitmaps from inline Data, not external SourceFile."""
    request = ExportRequest(
        output_dir=tmp_path,
        base_stem="embed",
        write_preview=False,
        write_lbrn2=True,
        write_pass_pngs=True,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    text = bundle.lbrn2_path.read_text(encoding="utf-8")
    # Embedded base64 PNG.
    assert "Data=" in text
    assert "iVBOR" in text  # base64 prefix of a PNG header
    # Project-level boilerplate that LightBurn requires to load layers.
    assert "<Thumbnail" in text
    assert "<VariableText>" in text
    assert "<UIPrefs>" in text


def test_export_without_lbrn2_skips_pass_stack(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """Default ExportRequest doesn't emit the pass stack."""
    request = ExportRequest(
        output_dir=tmp_path,
        base_stem="default",
        write_preview=False,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    assert bundle.lbrn2_path is None
    assert bundle.pass_png_paths == {}


def test_export_pass_pngs_without_lbrn2_persists_them(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """write_pass_pngs=True alone keeps the per-pass PNGs (no project)."""
    request = ExportRequest(
        output_dir=tmp_path,
        base_stem="pngs_only",
        write_preview=False,
        write_pass_pngs=True,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    assert bundle.lbrn2_path is None
    assert bundle.pass_png_paths
    for path in bundle.pass_png_paths.values():
        assert path.exists()


# ----------------------------------------------------------- API zip path

def test_api_export_lbrn2_returns_zip_with_project_and_pngs(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """The API path produces a self-contained zip — project + PNGs together."""
    from apps.api.service_adapter import do_export_lbrn2, store_plan
    from apps.api import blob_store as api_blob_store
    from mopa.stages import plan_passes

    request = ExportRequest(
        output_dir=tmp_path, base_stem="api_zip", write_preview=False,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    hm = np.asarray(Image.open(bundle.master16_png), dtype=np.float32) / 65535.0
    heightmap_id = api_blob_store.store_heightmap(hm)

    material = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    plan = plan_passes(heightmap=hm, profile=material)
    plan_id = store_plan(plan)

    zip_bytes = do_export_lbrn2(plan_id=plan_id, heightmap_id=heightmap_id)
    import io
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        assert "project.lbrn2" in names
        png_names = [n for n in names if n.startswith("pass_") and n.endswith(".png")]
        assert png_names, f"no per-pass PNGs in zip; saw {names}"

        # Regression — the API .lbrn2 was emitting empty <Shape Type="Bitmap">
        # tags with no SourceFile/Data, so LightBurn would open the project
        # but show blank layers. The fix is to embed the per-pass PNG bytes
        # as base64 Data on every Bitmap shape.
        lbrn2_text = zf.read("project.lbrn2").decode("utf-8")
        assert lbrn2_text.count("<Shape Type=\"Bitmap\"") == len(png_names), (
            "expected one Bitmap shape per pass PNG"
        )
        # Embedded PNG signature in base64 — proof the bitmap data made it
        # into the project file rather than being a stub reference.
        assert 'Data="iVBOR' in lbrn2_text, (
            "Bitmap shape has no embedded PNG Data — LightBurn won't render the layer"
        )
        # Each Bitmap shape needs a real W/H attribute too; without it the
        # transform collapses to zero-size and the bitmap is invisible.
        assert lbrn2_text.count(' W="') >= len(png_names)
        assert lbrn2_text.count(' H="') >= len(png_names)

        # Regression — the writer used to default to 50 mm on the longest
        # side regardless of heightmap resolution, which forced the user
        # to manually resize every bitmap in LightBurn. The export now
        # derives W/H from the heightmap pixel count at LightBurn's 120-DPI
        # natural-size convention (W = px × 25.4 / 120). The XForm scale
        # factor then maps from that natural size to the desired physical mm.
        # Pin that the size scales with the heightmap (≠ the old 50 mm
        # constant) and matches the 120-DPI rule for a known input.
        import re
        w_match = re.search(r' W="([0-9.]+)"', lbrn2_text)
        h_match = re.search(r' H="([0-9.]+)"', lbrn2_text)
        assert w_match and h_match
        w_mm = float(w_match.group(1))
        h_mm = float(h_match.group(1))
        # Heightmap is square in this fixture — both axes should match.
        assert abs(w_mm - h_mm) < 0.05, f"non-square output: {w_mm}×{h_mm}"
        # 120-DPI rule: W = px × 25.4 / 120 (LightBurn's natural-size DPI).
        # Confirmed against reference .lbrn2 file: 1920 px → W=406.4 mm.
        hm_arr = api_blob_store.load_heightmap(heightmap_id)
        assert hm_arr is not None
        expected_mm = hm_arr.shape[1] * 25.4 / 120.0
        assert abs(w_mm - expected_mm) < 0.05, (
            f"expected {expected_mm:.2f} mm at 120 DPI (LightBurn natural size) for "
            f"{hm_arr.shape[1]} px wide heightmap, got {w_mm}"
        )
        # Sanity — must have changed from the old 50 mm default for any
        # heightmap that isn't ~500 px wide.
        assert hm_arr.shape[1] != 500 or w_mm == 50.0, (
            "fixture happens to round-trip 50 mm; pick a different size"
        )


def test_api_export_lbrn2_fits_portrait_heightmap_into_profile_box_preserving_aspect():
    """When the profile defines a ``print_width_mm`` × ``print_height_mm``
    bounding box, the exporter must scale the heightmap into that box
    preserving its native aspect — never stretch a portrait into a
    square plaque profile. The embedded PNG bytes don't change; only
    the .lbrn2 W/H attributes do."""
    import io as _io
    import re
    import tempfile
    import yaml as _yaml
    from pathlib import Path
    from apps.api.service_adapter import do_export_lbrn2, store_plan
    from apps.api import blob_store as api_blob_store
    from mopa.stages import plan_passes
    from mopa.lightburn_cards import (
        DEFAULT_CARDS_DIR, DEFAULT_PROFILE_NAME, load_lightburn_card,
    )
    from mopa import profiles as _profiles

    # Heightmap 60×120 (0.5:1 portrait).
    hm = np.linspace(0.2, 0.9, 60 * 120, dtype=np.float32).reshape(120, 60)
    heightmap_id = api_blob_store.store_heightmap(hm)

    material = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    plan = plan_passes(heightmap=hm, profile=material)
    plan_id = store_plan(plan)

    # Synthesise a profile in a temp dir with a SQUARE 60 × 60
    # bounding box. A portrait heightmap should emerge as 30 × 60 mm
    # (fit-to-height; width is constrained well below the 60-mm box).
    tmp_user_dir = Path(tempfile.mkdtemp(prefix="mopa_profile_test_"))
    profile_path = tmp_user_dir / "test_square_box.yaml"
    profile_path.write_text(
        _yaml.safe_dump({
            "name": "test_square_box",
            "machine": "60W MOPA fiber",
            "lightburn_mode": "3D Sliced",
            "black_is_deep": True,
            "print_width_mm": 60.0,
            "print_height_mm": 60.0,
        }),
        encoding="utf-8",
    )
    # Point the loader at the temp dir via the documented env override
    # so we don't mutate the user's real ~/.mopa-heightmap/profiles.
    import os as _os
    saved_env = _os.environ.get(_profiles.USER_PROFILES_ENV)
    _os.environ[_profiles.USER_PROFILES_ENV] = str(tmp_user_dir)
    try:
        zip_bytes = do_export_lbrn2(
            plan_id=plan_id,
            heightmap_id=heightmap_id,
            profile_name="test_square_box",
        )
    finally:
        if saved_env is None:
            _os.environ.pop(_profiles.USER_PROFILES_ENV, None)
        else:
            _os.environ[_profiles.USER_PROFILES_ENV] = saved_env

    text = zipfile.ZipFile(_io.BytesIO(zip_bytes)).read("project.lbrn2").decode("utf-8")
    w_match = re.search(r' W="([0-9.]+)"', text)
    h_match = re.search(r' H="([0-9.]+)"', text)
    xform_match = re.search(r'<XForm>([^<]+)</XForm>', text)
    assert w_match and h_match and xform_match
    w_natural = float(w_match.group(1))
    h_natural = float(h_match.group(1))
    xform_vals = [float(v) for v in xform_match.group(1).split()]
    # W/H are the image's natural 120-DPI size; XForm a/d scale them to
    # physical mm. Physical display = W_natural × XForm_a.
    phys_w = w_natural * xform_vals[0]
    phys_h = h_natural * xform_vals[3]
    # Portrait heightmap (60×120) is padded to a square (120×120) before
    # embedding, so the physical output is sq_mm × sq_mm = 60×60 mm.
    assert abs(phys_w - 60.0) < 0.1, f"expected 60 mm physical width (square padded), got {phys_w}"
    assert abs(phys_h - 60.0) < 0.1, f"expected 60 mm physical height (square padded), got {phys_h}"
    # Image must be centred on the 175×175 mm bed (87.5, 87.5).
    assert abs(xform_vals[4] - 87.5) < 0.1, f"expected cx=87.5, got {xform_vals[4]}"
    assert abs(xform_vals[5] - 87.5) < 0.1, f"expected cy=87.5, got {xform_vals[5]}"


def test_api_export_lbrn2_subject_mask_baked_into_depth_png(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """Subject mask must be baked into per-pass depth PNGs at export time.

    Background pixels (mask=0) must become white (≈255) in the embedded PNG
    so the laser cannot fire there. LightBurn has no mechanism to clip via a
    raster PNG layer — only vector shapes (Ellipse MaskID) can do that, so
    the old output=0 M-layer approach provided zero laser guarantee.

    Verification: use a half-subject mask (top rows=255/subject, bottom=0/bg)
    on a uniform 0.5 heightmap. After export the depth PNG must have the
    background rows at ~255 (raised to surface) and subject rows below ~230
    (original depth ≈0.5 preserved)."""
    import base64 as _b64
    import io as _io
    import re
    import zipfile
    from apps.api.service_adapter import do_export_lbrn2, store_plan
    from apps.api import blob_store as api_blob_store
    from mopa.stages import plan_passes
    from mopa.lightburn_cards import DEFAULT_CARDS_DIR, DEFAULT_PROFILE_NAME, load_lightburn_card

    H, W = 64, 64
    hm = np.full((H, W), 0.5, dtype=np.float32)
    heightmap_id = api_blob_store.store_heightmap(hm)

    # Top half = subject (255), bottom half = background (0).
    mask_arr = np.zeros((H, W), dtype=np.uint8)
    mask_arr[:H // 2, :] = 255
    mbuf = _io.BytesIO()
    Image.fromarray(mask_arr, mode="L").save(mbuf, format="PNG")
    mask_id = api_blob_store.store_bytes(mbuf.getvalue(), content_type="image/png")

    material = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    plan = plan_passes(heightmap=hm, profile=material)
    plan_id = store_plan(plan)

    # Non-coin profile (no profile_name → no shape:circle) → baking path.
    zip_bytes = do_export_lbrn2(plan_id=plan_id, heightmap_id=heightmap_id,
                                subject_mask_id=mask_id)
    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as zf:
        lbrn2 = zf.read("project.lbrn2").decode("utf-8")

    # No M-layer: exactly one Bitmap shape per depth pass.
    n_passes = len(plan.passes)
    n_bitmaps = lbrn2.count('<Shape Type="Bitmap"')
    assert n_bitmaps == n_passes, (
        f"expected {n_passes} Bitmap shape(s), no M-layer; got {n_bitmaps}"
    )
    # No output=0 CutSetting anywhere.
    assert 'output Value="0"' not in lbrn2, "M-layer output=0 must not appear"

    # Decode the embedded PNG and verify baking.
    data_match = re.search(r'Data="([^"]+)"', lbrn2)
    assert data_match, "no embedded Data= in Bitmap shape"
    depth_img = Image.open(_io.BytesIO(_b64.b64decode(data_match.group(1)))).convert("L")
    depth_arr = np.asarray(depth_img, dtype=np.float32) / 255.0
    # The PNG is flipud before saving, so row 0 in the file = original bottom row.
    # Original bottom half = background (mask=0) → should be ~1.0 (white = no engrave).
    bg_rows = depth_arr[:H // 2, :]
    assert bg_rows.mean() > 0.95, (
        f"background rows should be ≈1.0 (no engraving); got mean={bg_rows.mean():.3f}"
    )
    # Original top half = subject (mask=1) → depth ≈0.5 preserved (< 0.9 after 8-bit quant).
    subject_rows = depth_arr[H // 2:, :]
    assert subject_rows.mean() < 0.9, (
        f"subject rows should preserve original ~0.5 depth; got mean={subject_rows.mean():.3f}"
    )


def test_api_export_lbrn2_coin_profile_uses_ellipse_not_baking():
    """Coin profiles (shape: circle) must use Ellipse MaskID for clipping,
    not pixel-baking. An all-black mask supplied to a coin export must NOT
    force the depth PNG to white — the Ellipse handles the boundary."""
    import base64 as _b64
    import io as _io
    import re
    import tempfile
    import yaml as _yaml
    import zipfile
    import os as _os
    from apps.api.service_adapter import do_export_lbrn2, store_plan
    from apps.api import blob_store as api_blob_store
    from mopa.stages import plan_passes
    from mopa.lightburn_cards import DEFAULT_CARDS_DIR, DEFAULT_PROFILE_NAME, load_lightburn_card
    from mopa import profiles as _profiles

    H, W = 64, 64
    hm = np.full((H, W), 0.5, dtype=np.float32)
    heightmap_id = api_blob_store.store_heightmap(hm)

    # All-black mask: if baking were applied, entire depth PNG would become white.
    mbuf = _io.BytesIO()
    Image.new("L", (W, H), color=0).save(mbuf, format="PNG")
    mask_id = api_blob_store.store_bytes(mbuf.getvalue(), content_type="image/png")

    material = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    plan = plan_passes(heightmap=hm, profile=material)
    plan_id = store_plan(plan)

    # Synthesise a circle profile in a temp dir.
    tmp_dir = Path(tempfile.mkdtemp(prefix="mopa_coin_test_"))
    (tmp_dir / "test_coin.yaml").write_text(
        _yaml.safe_dump({
            "name": "test_coin",
            "machine": "60W MOPA fiber",
            "lightburn_mode": "3D Sliced",
            "black_is_deep": True,
            "print_width_mm": 40.0,
            "print_height_mm": 40.0,
            "shape": "circle",
        }),
        encoding="utf-8",
    )
    saved = _os.environ.get(_profiles.USER_PROFILES_ENV)
    _os.environ[_profiles.USER_PROFILES_ENV] = str(tmp_dir)
    try:
        zip_bytes = do_export_lbrn2(
            plan_id=plan_id,
            heightmap_id=heightmap_id,
            profile_name="test_coin",
            subject_mask_id=mask_id,
        )
    finally:
        if saved is None:
            _os.environ.pop(_profiles.USER_PROFILES_ENV, None)
        else:
            _os.environ[_profiles.USER_PROFILES_ENV] = saved

    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as zf:
        lbrn2 = zf.read("project.lbrn2").decode("utf-8")

    # Ellipse present (coin clipping active).
    assert '<Shape Type="Ellipse"' in lbrn2, "coin export must have Ellipse"
    assert 'MaskID=' in lbrn2, "Bitmap must reference Ellipse via MaskID"

    # Depth PNG must NOT be forced to white — baking was skipped.
    data_match = re.search(r'Data="([^"]+)"', lbrn2)
    assert data_match
    depth_img = Image.open(_io.BytesIO(_b64.b64decode(data_match.group(1)))).convert("L")
    depth_arr = np.asarray(depth_img, dtype=np.float32) / 255.0
    # Original hm is 0.5 everywhere — if baking had run with all-black mask
    # every pixel would be 1.0. The coin path must preserve the ~0.5 depth.
    assert depth_arr.mean() < 0.9, (
        f"coin export must not bake the all-black mask into depth PNG; "
        f"mean={depth_arr.mean():.3f} (expected ~0.5)"
    )
