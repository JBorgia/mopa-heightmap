"""End-to-end .lbrn2 export integration test.

Exercises the full path the CLI / API rely on:

    HeightmapService.render -> derive_pass_masks -> plan_passes ->
    write_lbrn -> load_lightburn_card -> assert ColorEntry round-trips +
    per-pass PNG paths exist on disk + the project XML is well-formed
    LightBurn output (passes the same parser the importer uses for the
    canonical color cards).

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

from zoedepth.laser.lightburn_cards import (
    DEFAULT_CARDS_DIR,
    DEFAULT_PROFILE_NAME,
    load_lightburn_card,
)
from zoedepth.laser.service import (
    ExportRequest,
    HeightmapService,
    merge_profile_settings,
)
from zoedepth.laser.settings import AppSettings


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


def test_export_lbrn2_references_pngs_relatively(
    tmp_path: Path, service, settings_with_heightmap, synthetic_image,
):
    """SourceFile attributes must be relative paths so the bundle stays portable."""
    request = ExportRequest(
        output_dir=tmp_path,
        base_stem="rel",
        write_preview=False,
        write_lbrn2=True,
        write_pass_pngs=True,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    text = bundle.lbrn2_path.read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in text
    assert "SourceFile=" in text


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
    from zoedepth.laser.pass_masks import derive_pass_masks
    from zoedepth.laser.stages import plan_passes

    request = ExportRequest(
        output_dir=tmp_path, base_stem="api_zip", write_preview=False,
    )
    bundle = service.export(synthetic_image, settings_with_heightmap, request)
    hm = np.asarray(Image.open(bundle.master16_png), dtype=np.float32) / 65535.0
    heightmap_id = api_blob_store.store_heightmap(hm)

    material = load_lightburn_card(DEFAULT_CARDS_DIR / f"{DEFAULT_PROFILE_NAME}.lbrn2")
    plan = plan_passes(
        heightmap=hm, profile=material, masks=derive_pass_masks(hm),
    )
    plan_id = store_plan(plan)

    zip_bytes = do_export_lbrn2(plan_id=plan_id, heightmap_id=heightmap_id)
    import io
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        assert "project.lbrn2" in names
        png_names = [n for n in names if n.startswith("pass_") and n.endswith(".png")]
        assert png_names, f"no per-pass PNGs in zip; saw {names}"
