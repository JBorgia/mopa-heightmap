"""Live smoke render: king photo -> sculptok-style heightmap.

Throws the new subject_mask + relief stages at the existing ZoeDepth backbone
and saves the LightBurn-grade outputs next to the input. Nothing here is meant
to ship — it's a reproducible bench so we can compare against sculptok.png.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

from zoedepth.laser.profiles import load_profile
from zoedepth.laser.service import (
    ExportRequest,
    HeightmapService,
    InferenceConfig,
    merge_profile_settings,
)
from zoedepth.laser.settings import AppSettings


def main() -> int:
    here = Path(__file__).parent
    src = here / "assets" / "360_F_320748738_zddHlcaqbxBxOjXYpYpnQ4XlRT3cRS3H.jpg"
    out_dir = here / "outputs"
    out_dir.mkdir(exist_ok=True)

    profile = load_profile("sculptok_portrait")
    # Use BiRefNet for the cleanest silhouette (already cached, MIT). The
    # rest comes verbatim from the profile.
    settings = merge_profile_settings(profile, {
        "subject_mask_backend": "birefnet",
    })

    img = Image.open(src).convert("RGB")
    print(f"Source: {src.name}  ({img.size})")

    cfg = InferenceConfig(
        model_name="DAv2_Base",
        device=None,                    # auto
        inference_resolution=1024,      # DAv2 likes more pixels than ZoeDepth
        with_flip_aug=False,            # halves runtime
    )

    svc = HeightmapService(app_settings=AppSettings())
    t0 = time.perf_counter()
    bundle = svc.export(
        img,
        settings,
        cfg,
        ExportRequest(
            output_dir=out_dir,
            base_stem="king_sculptok",
            write_preview=True,
            naming="overwrite",
        ),
        profile_name="sculptok_portrait",
        profile_data=profile,
        input_path=str(src),
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"  preview:   {bundle.preview_png}")
    print(f"  lightburn: {bundle.lightburn_png}")
    print(f"  master16:  {bundle.master16_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
