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
        # derives mm from the heightmap pixel count at 254 DPI (≈10 px/mm).
        # Pin that the size scales with the heightmap (≠ the old 50 mm
        # constant) and matches the 254-DPI rule for a known input.
        import re
        w_match = re.search(r' W="([0-9.]+)"', lbrn2_text)
        h_match = re.search(r' H="([0-9.]+)"', lbrn2_text)
        assert w_match and h_match
        w_mm = float(w_match.group(1))
        h_mm = float(h_match.group(1))
        # Heightmap is square in this fixture — both axes should match.
        assert abs(w_mm - h_mm) < 0.05, f"non-square output: {w_mm}×{h_mm}"
        # 254-DPI rule: mm ≈ pixels / 10. The exact pixel count depends
        # on how the service rendered the input; we just verify the rule.
        hm_arr = api_blob_store.load_heightmap(heightmap_id)
        assert hm_arr is not None
        expected_mm = hm_arr.shape[1] * 25.4 / 254.0
        assert abs(w_mm - expected_mm) < 0.05, (
            f"expected {expected_mm:.2f} mm at 254 DPI for "
            f"{hm.shape[1]} px wide heightmap, got {w_mm}"
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
    assert w_match and h_match
    w_mm = float(w_match.group(1))
    h_mm = float(h_match.group(1))
    # Portrait heightmap (60×120) into 60×60 box → 30×60 mm. The box's
    # height is the binding constraint because the image is taller than
    # wide; width drops below the box maximum to preserve aspect.
    assert abs(w_mm - 30.0) < 0.01, f"expected ~30 mm width, got {w_mm}"
    assert abs(h_mm - 60.0) < 0.01, f"expected ~60 mm height, got {h_mm}"
    # Aspect must equal heightmap aspect — not the box aspect.
    assert abs((w_mm / h_mm) - (60.0 / 120.0)) < 1e-3


def test_api_export_lbrn2_with_subject_mask_adds_non_engraving_layer(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """Subject mask should arrive in the .lbrn2 as a SECOND Bitmap on a
    new CutSetting with output=0, so the user can toggle it in LightBurn
    without it firing on Start."""
    from apps.api.service_adapter import do_export_lbrn2, store_plan
    from apps.api import blob_store as api_blob_store
    from mopa.stages import plan_passes
    import io as _io
    import re
    from PIL import Image

    request = ExportRequest(
        output_dir=tmp_path, base_stem="api_mask", write_preview=False,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    hm = np.asarray(Image.open(bundle.master16_png), dtype=np.float32) / 65535.0
    heightmap_id = api_blob_store.store_heightmap(hm)

    material = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    plan = plan_passes(heightmap=hm, profile=material)
    plan_id = store_plan(plan)

    # Synthesise a 1×1 mask blob — the contents don't matter, just that
    # the export inlines it as a layer.
    mbuf = _io.BytesIO()
    Image.new("L", (4, 4), color=255).save(mbuf, format="PNG")
    mask_id = api_blob_store.store_bytes(mbuf.getvalue(), content_type="image/png")

    zip_bytes = do_export_lbrn2(
        plan_id=plan_id, heightmap_id=heightmap_id, subject_mask_id=mask_id,
    )
    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        # The mask is embedded as base64 inside project.lbrn2 (verified
        # below), so the inner .lbrn2 zip MUST NOT also write it as a
        # standalone file — that would ship the same bytes twice in
        # /export/bundle (the bundle endpoint adds the standalone copy
        # separately from blob_store).
        assert "subject_mask.png" not in names, (
            f"mask should be embedded only, not duplicated as a file; saw {names}"
        )
        text = zf.read("project.lbrn2").decode("utf-8")
        # Two Bitmap shapes: one depth pass + one mask.
        assert text.count('<Shape Type="Bitmap"') == 2
        # The mask CutSetting must have output=0 so LightBurn won't fire it
        # on Start. Find it by scanning for the CutSetting_Img blocks.
        cut_settings = re.findall(
            r'<CutSetting_Img[^>]*>.*?</CutSetting_Img>', text, flags=re.DOTALL,
        )
        # At least one CutSetting must have output Value="0" — the mask one.
        outputs = [
            re.search(r'<output Value="(\d)"', cs)
            for cs in cut_settings
        ]
        non_engraving = [m.group(1) for m in outputs if m and m.group(1) == "0"]
        assert non_engraving, (
            "expected a CutSetting with output=0 (the non-engraving mask "
            "layer); none found"
        )
        # The mask layer should also be named so users can spot it in the
        # LightBurn layer list.
        assert "subject_mask" in text or "M99" in text or "M100" in text

        # Regression — LightBurn 1.7 crashes when a CutSetting_Img block
        # is missing required fields (bidir, priority, tabCount,
        # tabCountMax) or has invalid values (numPasses=0, non-standard
        # subname). The mask layer's CutSetting is built by cloning the
        # depth pass's structure, so both must carry the same field set.
        def _tags(block: str) -> set[str]:
            return set(re.findall(r"<(\w+) Value=\"[^\"]*\"/>", block))

        depth_block = next(cs for cs in cut_settings if 'name Value="C01"' in cs)
        mask_block = next(cs for cs in cut_settings if 'name Value="M' in cs)
        depth_tags = _tags(depth_block)
        mask_tags = _tags(mask_block)
        # Mask must have every field the depth cut has (plus possibly extras
        # like ``output`` which the depth doesn't bother setting).
        missing = depth_tags - mask_tags
        assert not missing, (
            f"mask CutSetting missing fields {missing} that LightBurn requires; "
            f"depth has {depth_tags}, mask has {mask_tags}"
        )
        # numPasses must be > 0 — LightBurn rejects 0-pass layers.
        np_match = re.search(r'<numPasses Value="(\d+)"', mask_block)
        assert np_match and int(np_match.group(1)) >= 1, (
            f"mask numPasses must be ≥ 1 (LightBurn crashes on 0); got {mask_block}"
        )
        # subname must be a known LightBurn value, not a custom string.
        subname_match = re.search(r'<subname Value="([^"]+)"', mask_block)
        assert subname_match and subname_match.group(1) in {
            "3D Slice", "Image", "Fill", "Line",
        }, f"unknown subname {subname_match!r}; LightBurn only knows the standard set"

        # XForm scales must match the depth pass — the mask is resized to
        # the heightmap pixel dims so both shapes share an identical scale
        # factor. Without this the mask renders stretched in LightBurn
        # because the writer computes per-shape scale from each PNG's own
        # pixel count and the source-photo mask has a different aspect
        # than the sculptok heightmap.
        depth_shape = re.search(
            r'<Shape Type="Bitmap" CutIndex="\d+"[^>]*>\s*<XForm>([^<]+)</XForm>',
            text,
        )
        all_shapes = re.findall(
            r'<Shape Type="Bitmap" CutIndex="(\d+)"[^>]*>\s*<XForm>([^<]+)</XForm>',
            text,
        )
        assert len(all_shapes) == 2
        # Parse the six XForm values; sx and -sy must be equal across shapes.
        scales = [tuple(float(v) for v in xform.split()) for _, xform in all_shapes]
        assert abs(scales[0][0] - scales[1][0]) < 1e-6, (
            f"XForm sx mismatch: {scales} — mask not resized to heightmap dims"
        )
        assert abs(scales[0][3] - scales[1][3]) < 1e-6, (
            f"XForm sy mismatch: {scales} — mask not resized to heightmap dims"
        )
