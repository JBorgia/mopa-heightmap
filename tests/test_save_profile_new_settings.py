"""Round-trip test: scaffold a profile carrying the new heightmap keys
(face_relief, depth_unsharp, subject_mask, relief, signature, …) and
confirm it loads back through ``load_profile`` with every key intact.

This guards against the profile validator drifting away from the
DEFAULT_SETTINGS engine surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zoedepth.laser.profiles import load_profile, scaffold_profile
from zoedepth.laser.service import DEFAULT_SETTINGS


def test_scaffold_with_new_settings_round_trips(tmp_path: Path):
    custom = dict(DEFAULT_SETTINGS)
    custom.update({
        "subject_mask_enabled": True,
        "subject_mask_backend": "rembg",
        "subject_mask_feather_px": 5,
        "subject_mask_threshold": 0.6,
        "relief_enabled": True,
        "relief_strength": 0.4,
        "relief_normals_backend": "finite_diff",
        "depth_unsharp_enabled": True,
        "depth_unsharp_gamma": 0.65,
        "depth_unsharp_blend": 0.35,
        "face_relief_enabled": True,
        "face_relief_strength": 1.2,
        "auto_orient_face": True,
        "delight_enabled": True,
        "delight_backend": "marigold_iid",
        "depth_bilateral_enabled": True,
        "depth_bilateral_diameter": 11,
        "depth_bilateral_sigma_color": 0.07,
        "depth_bilateral_sigma_space": 9.0,
        "signature_text": "JB 2026",
        "signature_corner": "tr",
        "signature_height_fraction": 0.05,
        "signature_margin_fraction": 0.02,
    })

    path = scaffold_profile(
        "_test_new_settings",
        custom,
        target_dir=tmp_path,
        overwrite=True,
    )
    assert path.exists()

    loaded = load_profile(str(path))
    block = loaded.get("heightmap", {})

    # Every customised key must round-trip.
    for key, expected in custom.items():
        if expected == DEFAULT_SETTINGS.get(key):
            # scaffold_profile strips fields equal to engine default; that's by design.
            continue
        assert key in block, f"{key} missing from saved profile"
        assert block[key] == expected, (
            f"{key} drifted on round-trip: {expected!r} -> {block[key]!r}"
        )


def test_save_profile_rejects_invalid_signature_corner(tmp_path: Path):
    custom = dict(DEFAULT_SETTINGS)
    custom["signature_corner"] = "middle"   # not a real corner
    custom["signature_text"] = "x"
    with pytest.raises(Exception):
        scaffold_profile(
            "_test_bad_corner",
            custom,
            target_dir=tmp_path,
            overwrite=True,
        )


def test_save_profile_accepts_empty_signature_text(tmp_path: Path):
    custom = dict(DEFAULT_SETTINGS)
    custom["signature_text"] = ""           # the "off" sentinel
    path = scaffold_profile(
        "_test_empty_signature",
        custom,
        target_dir=tmp_path,
        overwrite=True,
    )
    loaded = load_profile(str(path))
    # Empty string is the engine default, so scaffold_profile may strip it,
    # but the profile must still load without errors.
    assert "heightmap" in loaded
