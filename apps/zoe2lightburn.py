from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from zoedepth.laser.preview import create_calibration_ramp
from zoedepth.laser.profiles import load_profile
from zoedepth.laser.service import (
    DEFAULT_SETTINGS,
    ExportRequest,
    HeightmapService,
    InferenceConfig,
    merge_profile_settings,
)
from zoedepth.laser.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ZoeDepth output into LightBurn 3D Sliced-ready heightmaps."
    )
    parser.add_argument("input", nargs="*", help="Input image path(s); pass multiple for batch")
    parser.add_argument("--output", help="Output PNG path for the 8-bit LightBurn heightmap")
    parser.add_argument("--profile", help="Material profile name or YAML path")
    parser.add_argument(
        "--model",
        default=None,
        help="Depth backend key — ZoeD_N, ZoeD_K, ZoeD_NK, DAv2_Small, DAv2_Base, "
             "DAv2_Large (CC-BY-NC), or Sapiens_Depth_1B (CC-BY-NC). Backends "
             "are looked up in zoedepth.laser.backends; unknown keys fall back "
             "to hubconf entrypoints.",
    )
    parser.add_argument("--device", default=None, help="Inference device, for example cpu or cuda")
    parser.add_argument("--precision", default=None, choices=["auto", "fp32", "fp16", "bf16"],
                        help="Mixed-precision mode for ZoeDepth on CUDA (auto = fp16 on GPU, fp32 on CPU)")
    parser.add_argument("--inference-resolution", dest="inference_resolution", type=int, default=None,
                        help="Cap longest side fed to ZoeDepth (0 = full). 256/384/512 trade quality for speed.")
    parser.add_argument("--make-ramp", help="Write a standalone calibration ramp PNG and exit")
    parser.add_argument("--export-preview", action="store_true", help="Write a preview PNG beside the LightBurn output")
    parser.add_argument("--export-calibration-ramp", action="store_true", help="Write a calibration ramp beside the LightBurn output")
    parser.add_argument(
        "--export-lbrn2",
        action="store_true",
        help="Also write a LightBurn .lbrn2 project + per-pass PNGs ready to open in LightBurn",
    )
    parser.add_argument(
        "--write-pass-pngs",
        action="store_true",
        help="Persist the per-pass PNGs alongside the .lbrn2 (otherwise they're "
             "embedded in the project but cleaned up after).",
    )
    parser.add_argument(
        "--lightburn-card",
        default=None,
        help="Name of the LightBurn color card to lift cut settings from "
             "(e.g. Colour60W-M7). Defaults to the active material profile or the shipped 60W card.",
    )
    parser.add_argument(
        "--n-color-passes",
        type=int,
        default=0,
        help="Number of MOPA color passes to derive from the photo via LAB k-means. "
             "0 = monochrome stack (default).",
    )
    parser.add_argument(
        "--disable-pass",
        dest="disable_passes",
        action="append",
        default=[],
        help="Disable a specific pass kind in the engraving stack "
             "(form, cleanup, detail, shading, polish, signature, pre_clean). Repeatable.",
    )
    parser.add_argument(
        "--export-clb",
        action="store_true",
        help="Also write a LightBurn Cut Library (.clb) file with every cut "
             "setting from the active material card.",
    )
    parser.add_argument(
        "--print-width-mm",
        dest="print_width_mm",
        type=float,
        default=None,
        help="Physical print width in mm for burn-time estimation. "
             "Defaults to 50 mm on the longest heightmap dimension.",
    )
    parser.add_argument(
        "--print-height-mm",
        dest="print_height_mm",
        type=float,
        default=None,
        help="Physical print height in mm for burn-time estimation.",
    )
    parser.add_argument(
        "--auto-orient-face",
        dest="auto_orient_face",
        action="store_true",
        default=None,
        help="Rotate the input so the inter-pupillary line is level. "
             "No-op when no face is detected.",
    )
    parser.add_argument(
        "--delight",
        dest="delight_enabled",
        action="store_true",
        default=None,
        help="Run Marigold-IID-Appearance delighting before depth (CC-BY-NC-4.0).",
    )
    parser.add_argument(
        "--depth-bilateral",
        dest="depth_bilateral_enabled",
        action="store_true",
        default=None,
        help="Photo-guided bilateral cross-filter on the raw depth (sharpens hair / "
             "fabric edges to photo edges).",
    )
    parser.add_argument(
        "--signature",
        dest="signature_text",
        default=None,
        help='Render this text as a small relief signature in a corner '
             '(e.g. "JB 2026"). Empty disables.',
    )
    parser.add_argument(
        "--signature-corner",
        dest="signature_corner",
        choices=["tl", "tr", "bl", "br"],
        default=None,
        help="Which corner to place the signature in. Default: br.",
    )
    parser.add_argument(
        "--heightmap",
        dest="external_heightmap_path",
        default=None,
        help="Path to a precomputed heightmap PNG (sculptok / meshy / hand-authored). "
             "When set, the depth network is bypassed and this image is used as the "
             "depth source. Pair with the original photo for subject mask + colour "
             "passes + LightBurn export.",
    )
    parser.add_argument(
        "--heightmap-polarity",
        dest="external_heightmap_polarity",
        choices=["bright_raised", "dark_raised", "auto"],
        default=None,
        help="Polarity of --heightmap input. Default: bright_raised (sculptok / meshy / "
             "most published bas-relief renders). Use 'dark_raised' if your tool emits "
             "the inverted polarity, or 'auto' to sniff the corners.",
    )
    parser.add_argument(
        "--no-heightmap-stretch",
        dest="external_heightmap_auto_stretch",
        action="store_false",
        default=None,
        help="Skip the auto-stretch of the in-subject range. Use when your --heightmap "
             "input is already calibrated to the engraving budget.",
    )
    parser.add_argument(
        "--no-heightmap-mask",
        dest="external_heightmap_use_subject_mask",
        action="store_false",
        default=None,
        help="Skip the BiRefNet subject mask + background flatten on --heightmap input. "
             "Use when your supplied heightmap already has a clean cutout.",
    )
    parser.add_argument(
        "--use-sculptok",
        action="store_true",
        help="Send the photo to the Sculptok API, wait for the heightmap, then "
             "feed it through --heightmap mode. Requires a Sculptok API key (see "
             "--sculptok-api-key / SCULPTOK_API_KEY / settings.json).",
    )
    parser.add_argument(
        "--sculptok-api-key",
        dest="sculptok_api_key",
        default=None,
        help="Sculptok API key. Overrides SCULPTOK_API_KEY env var and settings.json. "
             "Avoid passing on the command line in production (visible in shell history).",
    )
    parser.add_argument(
        "--sculptok-style",
        dest="sculptok_style",
        choices=["normal", "portrait", "sketch", "pro"],
        default="pro",
        help="Sculptok depth-map style. 'pro' (default) costs 15-30 credits but gives "
             "the best detail; the others cost 10 credits.",
    )
    parser.add_argument(
        "--sculptok-version",
        dest="sculptok_version",
        choices=["1.0", "1.5"],
        default="1.5",
        help="Sculptok pro-model version. 1.5 is the newer release (default).",
    )
    parser.add_argument(
        "--sculptok-hd",
        dest="sculptok_draw_hd",
        choices=["2k", "4k"],
        default="2k",
        help="Sculptok pro-model resolution. 4k doubles the credit cost (15 → 30).",
    )
    parser.add_argument("--tile-size", type=int, default=0, help="Optional tile size for large images; 0 disables tiled inference")
    parser.add_argument("--tile-overlap", type=int, default=128, help="Overlap in pixels for tiled inference")
    parser.add_argument("--pad-input", dest="pad_input", action="store_true", default=None, help="Enable ZoeDepth padding augmentation")
    parser.add_argument("--no-pad-input", dest="pad_input", action="store_false", help="Disable ZoeDepth padding augmentation")
    parser.add_argument("--with-flip-aug", dest="with_flip_aug", action="store_true", default=None, help="Enable ZoeDepth horizontal flip augmentation")
    parser.add_argument("--no-flip-aug", dest="with_flip_aug", action="store_false", help="Disable ZoeDepth horizontal flip augmentation")

    parser.add_argument("--naming", choices=["overwrite", "timestamp", "counter"], default=None, help="Output file naming policy")
    parser.add_argument("--keep-history", dest="keep_history", action="store_true", default=None, help="Never overwrite previous exports")
    parser.add_argument("--gui", action="store_true", help="Launch the Gradio Studio UI")

    parser.add_argument("--near", dest="near_percentile", type=float, default=None)
    parser.add_argument("--far", dest="far_percentile", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--contrast", type=float, default=None)
    parser.add_argument("--midtone-boost", dest="midtone_boost", type=float, default=None)
    parser.add_argument("--deep-limit", dest="deep_limit", type=float, default=None)
    parser.add_argument("--surface-limit", dest="surface_limit", type=float, default=None)
    parser.add_argument("--smooth", choices=["none", "off", "bilateral", "gaussian"], default=None)
    parser.add_argument("--smooth-diameter", dest="smooth_diameter", type=int, default=None)
    parser.add_argument("--smooth-strength", dest="smooth_strength", type=float, default=None)
    parser.add_argument("--sharpen", type=float, default=None)
    parser.add_argument("--sharpen-sigma", dest="sharpen_sigma", type=float, default=None)

    parser.add_argument("--flatten-background", dest="flatten_background", action="store_true", default=None)
    parser.add_argument("--no-flatten-background", dest="flatten_background", action="store_false")
    parser.add_argument("--background-threshold", dest="background_threshold", type=float, default=None)
    parser.add_argument("--background-value", dest="background_value", type=float, default=None)

    polarity = parser.add_mutually_exclusive_group()
    polarity.add_argument("--black-is-deep", dest="black_is_deep", action="store_true")
    polarity.add_argument("--white-is-deep", dest="black_is_deep", action="store_false")
    polarity.add_argument("--negative-compatible", dest="black_is_deep", action="store_false", help="Alias for --white-is-deep when using LightBurn Negative Image")
    parser.set_defaults(black_is_deep=None)

    # Stage A — input conditioning toggles.
    parser.add_argument("--white-balance", dest="input_white_balance", action="store_true", default=None)
    parser.add_argument("--clahe", dest="input_clahe", action="store_true", default=None)
    parser.add_argument("--clahe-clip", dest="input_clahe_clip", type=float, default=None)
    parser.add_argument("--clahe-grid", dest="input_clahe_grid", type=int, default=None)
    parser.add_argument("--denoise", dest="input_denoise", action="store_true", default=None)
    parser.add_argument("--denoise-strength", dest="input_denoise_strength", type=float, default=None)
    parser.add_argument("--remove-specular", dest="input_remove_specular", action="store_true", default=None)
    parser.add_argument("--specular-threshold", dest="input_specular_threshold", type=int, default=None)
    parser.add_argument("--max-input-dim", dest="input_max_dim", type=int, default=None)
    # Stage C extras.
    parser.add_argument("--edge-refine", dest="edge_refine", action="store_true", default=None)
    parser.add_argument("--edge-refine-diameter", dest="edge_refine_diameter", type=int, default=None)
    parser.add_argument("--edge-refine-sigma-color", dest="edge_refine_sigma_color", type=float, default=None)
    parser.add_argument("--edge-refine-sigma-space", dest="edge_refine_sigma_space", type=float, default=None)
    parser.add_argument("--dither", dest="dither", action="store_true", default=None)
    parser.add_argument("--dither-levels", dest="dither_levels", type=int, default=None)
    parser.add_argument("--target-depth-um", dest="target_depth_um", type=float, default=None,
                        help="Target physical depth (µm) for LUT remapping; 0 uses the profile's LUT max.")
    parser.add_argument("--posterize", dest="posterize_passes", type=int, default=None,
                        help="Quantize output to N pass-levels for an engraver-accurate preview (0 = off).")
    # Re-run from a sidecar JSON.
    parser.add_argument(
        "--rerun-from-settings",
        dest="rerun_from_settings",
        help="Replay a previous export using its *_settings.json sidecar.",
    )
    # Profile authoring.
    parser.add_argument(
        "--save-profile",
        dest="save_profile",
        help="Write the current settings as a new user-scope profile and exit.",
    )
    parser.add_argument(
        "--save-profile-overwrite",
        dest="save_profile_overwrite",
        action="store_true",
        help="Allow --save-profile to overwrite an existing file.",
    )
    return parser


def _heightmap_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: getattr(args, key, None) for key in DEFAULT_SETTINGS.keys()}


def _resolve_output(input_path: Path, output: str | None, output_dir_default: str) -> tuple[Path, str]:
    if output:
        out_path = Path(output)
        directory = out_path.parent if str(out_path.parent) not in {"", "."} else Path(output_dir_default)
        return directory, out_path.stem
    return Path(output_dir_default), input_path.stem


def _print_bundle(bundle) -> None:
    print(f"Saved LightBurn heightmap to {bundle.lightburn_png}")
    print(f"Saved 16-bit master to {bundle.master16_png}")
    if bundle.preview_png:
        print(f"Saved preview to {bundle.preview_png}")
    if bundle.ramp_png:
        print(f"Saved calibration ramp to {bundle.ramp_png}")
    print(f"Saved settings to {bundle.settings_json}")
    if bundle.lbrn2_path:
        print(f"Saved LightBurn project to {bundle.lbrn2_path}")
    if bundle.clb_path:
        print(f"Saved Cut Library to {bundle.clb_path}")
    if bundle.pass_png_paths:
        print(f"Saved {len(bundle.pass_png_paths)} per-pass PNG(s):")
        for kind, path in bundle.pass_png_paths.items():
            print(f"  {kind}: {path.name}")
    if bundle.burn_estimate:
        be = bundle.burn_estimate
        print(f"Burn estimate ({be['width_mm']:.0f} × {be['height_mm']:.0f} mm): "
              f"{be['total_pretty']} total over {len(be['passes'])} pass(es)")
    if bundle.qa_findings:
        print(f"QA: {len(bundle.qa_findings)} finding(s):")
        for f in bundle.qa_findings:
            print(f"  [{f['severity']}] {f['code']}: {f['message']}")
    print(f"Done in {bundle.elapsed_s:.2f}s")


def _run_rerun(args: argparse.Namespace, app_settings) -> None:
    from zoedepth.laser.rerun import payload_from_sidecar, request_for_sidecar

    payload = payload_from_sidecar(args.rerun_from_settings)
    inputs = [Path(p) for p in args.input] if args.input else (
        [payload.input_path] if payload.input_path else []
    )
    if not inputs:
        raise SystemExit("--rerun-from-settings requires at least one input image.")

    service = HeightmapService(app_settings=app_settings)
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        output_dir, stem = _resolve_output(input_path, args.output, app_settings.output.directory)
        request = request_for_sidecar(
            payload, output_dir, stem,
            naming=args.naming or "counter",
            write_preview=args.export_preview or True,
        )
        bundle = service.export(
            Image.open(input_path),
            payload.settings,
            payload.inference,
            request,
            profile_name=payload.profile_name,
            profile_data=payload.profile_data,
            input_path=input_path,
        )
        _print_bundle(bundle)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui:
        try:
            from ui.mopa_studio import launch
        except ImportError as exc:
            raise SystemExit(f"GUI is unavailable: {exc}") from exc
        launch()
        return

    if args.make_ramp:
        ramp_path = Path(args.make_ramp)
        ramp_path.parent.mkdir(parents=True, exist_ok=True)
        create_calibration_ramp().save(ramp_path)
        print(f"Saved calibration ramp to {ramp_path}")
        return

    app_settings = load_settings()

    if args.rerun_from_settings:
        _run_rerun(args, app_settings)
        return

    if args.save_profile:
        from zoedepth.laser.profiles import scaffold_profile

        profile_data: Dict[str, Any] = {}
        if args.profile:
            profile_data = load_profile(args.profile)
        overrides = _heightmap_overrides(args)
        merged = merge_profile_settings(profile_data, overrides)
        path = scaffold_profile(
            args.save_profile,
            merged,
            black_is_deep=bool(merged.get("black_is_deep", True)),
            lightburn_starting_point=profile_data.get("lightburn_starting_point"),
            overwrite=args.save_profile_overwrite,
        )
        print(f"Saved profile to {path}")
        return

    if not args.input:
        parser.error("input is required unless --make-ramp, --gui, --rerun-from-settings, or --save-profile is used")

    inputs = [Path(p) for p in args.input]
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

    profile_data: Dict[str, Any] = {}
    if args.profile:
        profile_data = load_profile(args.profile)

    overrides = _heightmap_overrides(args)
    settings = merge_profile_settings(profile_data, overrides)

    # --use-sculptok auto-pull: configure the client up-front so we can
    # do a single balance check before burning any credits. The actual
    # API calls happen inside the per-input render loop below so each
    # input gets its own freshly-generated heightmap (multi-input runs
    # don't accidentally share one heightmap across photos).
    sculptok_client = None
    sculptok_params = None
    if getattr(args, "use_sculptok", False):
        from zoedepth.laser.settings import resolve_sculptok_api_key
        from zoedepth.laser.sculptok_client import (
            SculptokClient, SculptokDepthMapParams,
        )

        api_key = resolve_sculptok_api_key(
            cli_value=args.sculptok_api_key, settings=app_settings,
        )
        if not api_key:
            raise SystemExit(
                "Sculptok API key not configured. Set SCULPTOK_API_KEY env var, "
                "pass --sculptok-api-key, or add credentials.sculptok_api_key to "
                "~/.mopa-heightmap/settings.json."
            )
        sculptok_client = SculptokClient(api_key)
        sculptok_params = SculptokDepthMapParams(
            style=args.sculptok_style,
            version=args.sculptok_version,
            draw_hd=args.sculptok_draw_hd,
        )
        balance = sculptok_client.get_credits()
        cost = sculptok_params.expected_cost()
        print(
            f"Sculptok credits: {balance} (each {sculptok_params.style}/"
            f"{sculptok_params.draw_hd} call costs {cost})"
        )
        if balance < cost * len(inputs):
            raise SystemExit(
                f"Need {cost * len(inputs)} credits for {len(inputs)} input(s); "
                f"only {balance} available. Top up at https://www.sculptok.com/pricing."
            )

    inference_cfg = InferenceConfig(
        model_name=args.model or app_settings.inference.default_model,
        device=args.device,
        pad_input=app_settings.inference.pad_input if args.pad_input is None else args.pad_input,
        with_flip_aug=app_settings.inference.flip_aug if args.with_flip_aug is None else args.with_flip_aug,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        precision=args.precision,
        inference_resolution=args.inference_resolution if args.inference_resolution is not None else app_settings.inference.inference_resolution,
    )

    service = HeightmapService(app_settings=app_settings)
    multi = len(inputs) > 1
    for index, input_path in enumerate(inputs):
        # In batch mode, --output is interpreted as a directory only.
        if multi and args.output:
            output_dir, stem = Path(args.output), input_path.stem
        else:
            output_dir, stem = _resolve_output(input_path, args.output, app_settings.output.directory)

        pass_toggles = {kind: False for kind in args.disable_passes}
        request = ExportRequest(
            output_dir=output_dir,
            base_stem=stem,
            write_preview=args.export_preview,
            write_calibration_ramp=args.export_calibration_ramp and index == 0,
            naming=args.naming or app_settings.output.naming,
            timestamp_format=app_settings.output.timestamp_format,
            keep_history=app_settings.output.keep_history if args.keep_history is None else bool(args.keep_history),
            write_lbrn2=bool(args.export_lbrn2),
            write_pass_pngs=bool(args.write_pass_pngs or args.export_lbrn2),
            write_clb=bool(args.export_clb),
            lightburn_card=args.lightburn_card,
            n_color_passes=int(args.n_color_passes or 0),
            pass_toggles=pass_toggles,
            print_width_mm=args.print_width_mm,
            print_height_mm=args.print_height_mm,
        )

        if multi:
            print(f"[{index + 1}/{len(inputs)}] {input_path}")

        # Per-input Sculptok call: produces a heightmap PNG sibling of
        # the photo and points this run's external_heightmap_path at it.
        per_input_settings = dict(settings)
        if sculptok_client is not None and sculptok_params is not None:
            from zoedepth.laser.sculptok_client import SculptokInsufficientCreditsError

            print(
                f"  Sculptok: generating depth-map "
                f"({sculptok_params.style}/{sculptok_params.draw_hd}, "
                f"{sculptok_params.expected_cost()} credits) ..."
            )
            try:
                heightmap_path = sculptok_client.generate_heightmap(
                    input_path,
                    params=sculptok_params,
                    out_path=input_path.with_name(input_path.stem + "_sculptok.png"),
                    check_credits=False,    # checked once before the loop
                    on_status=lambda s: print(
                        f"    ... task {s.prompt_id[:8]} status={s.status} "
                        f"step={s.current_step} queue={s.queue_position}",
                        flush=True,
                    ),
                )
            except SculptokInsufficientCreditsError as exc:
                raise SystemExit(f"Sculptok: {exc}") from exc
            print(f"  Sculptok: {heightmap_path}")
            per_input_settings["external_heightmap_path"] = str(heightmap_path)

        bundle = service.export(
            Image.open(input_path),
            per_input_settings,
            inference_cfg,
            request,
            profile_name=args.profile,
            profile_data=profile_data,
            input_path=input_path,
        )
        _print_bundle(bundle)


if __name__ == "__main__":
    main()
