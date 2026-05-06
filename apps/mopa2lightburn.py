"""CLI: photo + sculptok heightmap → LightBurn 3D Sliced bundle.

Two ways to supply the heightmap:

    --heightmap <path>     bring-your-own PNG (sculptok / meshy / hand-authored)
    --use-sculptok         auto-pull from the Sculptok API (consumes credits)

The CLI then produces a LightBurn-ready bundle: heightmap PNG, .lbrn2
project, .clb cut library, per-pass PNGs, optional preview, and a
settings.json sidecar.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from mopa.preview import create_calibration_ramp
from mopa.profiles import load_profile
from mopa.service import (
    DEFAULT_SETTINGS,
    ExportRequest,
    HeightmapService,
    merge_profile_settings,
)
from mopa.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a photo + sculptok heightmap into a LightBurn 3D Sliced bundle."
    )
    parser.add_argument("input", nargs="*", help="Input image path(s); pass multiple for batch.")
    parser.add_argument(
        "--output",
        help="Parent directory for the export bundle. Each export creates a "
             "<output>/<bundle_name>/ folder containing 'final/' (drag-into-LightBurn) "
             "and 'work/' (preview, settings, sources). Defaults to ./outputs/.",
    )
    parser.add_argument(
        "--name",
        dest="bundle_name",
        default=None,
        help="Override the bundle folder name (defaults to the input image's filename stem).",
    )
    parser.add_argument("--profile", help="Material profile name or YAML path")
    parser.add_argument(
        "--target",
        help="Target-object preset (coin / signet_ring / pendant / plaque / portrait) "
             "or path to a target YAML. Sets print dimensions, polarity-invert default, "
             "and a starter heightmap-settings block.",
    )
    parser.add_argument("--make-ramp", help="Write a standalone calibration ramp PNG and exit")

    # ----------------------------------------- heightmap source (one required)
    parser.add_argument(
        "--heightmap",
        dest="external_heightmap_path",
        default=None,
        help="Path to a precomputed heightmap PNG (sculptok / meshy / hand-authored).",
    )
    parser.add_argument(
        "--heightmap-polarity",
        dest="external_heightmap_polarity",
        choices=["bright_raised", "dark_raised", "auto"],
        default=None,
        help="Polarity of --heightmap input. Default: bright_raised (sculptok / meshy).",
    )
    parser.add_argument(
        "--polarity-invert",
        dest="polarity_invert",
        action="store_true",
        default=None,
        help="Flip the heightmap so the subject engraves deep instead of the background. "
             "Use for signet rings and other recessed designs.",
    )

    # ---------------------------------------------- sculptok auto-pull
    parser.add_argument(
        "--use-sculptok",
        action="store_true",
        help="Send the photo to the Sculptok API, wait for the heightmap, and use it. "
             "Requires a Sculptok API key (--sculptok-api-key, SCULPTOK_API_KEY, or settings.json).",
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
        help="Sculptok pro-model resolution. 4k doubles the credit cost (15 -> 30).",
    )

    # ----------------------------------------------- export deliverables
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
        "--export-clb",
        action="store_true",
        help="Also write a LightBurn Cut Library (.clb) file with every cut "
             "setting from the active material card.",
    )
    parser.add_argument(
        "--export-mask",
        dest="write_subject_mask",
        action="store_true",
        help="Write a separate subject mask PNG to the bundle's final/ folder.",
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
        help="Disable a specific pass kind in the engraving stack (repeatable).",
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

    # ------------------------------------------- pre-sculptok input prep
    parser.add_argument("--white-balance", dest="input_white_balance", action="store_true", default=None)
    parser.add_argument("--clahe", dest="input_clahe", action="store_true", default=None)
    parser.add_argument("--clahe-clip", dest="input_clahe_clip", type=float, default=None)
    parser.add_argument("--clahe-grid", dest="input_clahe_grid", type=int, default=None)
    parser.add_argument("--denoise", dest="input_denoise", action="store_true", default=None)
    parser.add_argument("--denoise-strength", dest="input_denoise_strength", type=float, default=None)
    parser.add_argument("--remove-specular", dest="input_remove_specular", action="store_true", default=None)
    parser.add_argument("--specular-threshold", dest="input_specular_threshold", type=int, default=None)
    parser.add_argument("--max-input-dim", dest="input_max_dim", type=int, default=None)

    # ------------------------------------------- subject mask deliverable
    parser.add_argument("--subject-mask", dest="subject_mask_enabled", action="store_true", default=None)
    parser.add_argument("--subject-mask-backend", dest="subject_mask_backend", default=None)
    parser.add_argument("--subject-mask-feather", dest="subject_mask_feather_px", type=int, default=None)
    parser.add_argument("--subject-mask-threshold", dest="subject_mask_threshold", type=float, default=None)

    # ----------------------------------------------------- photo-tonal pass
    parser.add_argument("--photo-tonal", dest="photo_tonal_enabled", action="store_true", default=None)
    parser.add_argument("--photo-tonal-strength", dest="photo_tonal_strength", type=float, default=None)
    parser.add_argument("--photo-tonal-invert", dest="photo_tonal_invert", action="store_true", default=None)

    # --------------------------------------------------------- signature
    parser.add_argument("--signature", dest="signature_text", default=None,
                        help='Render this text as a small relief signature in a corner.')
    parser.add_argument("--signature-corner", dest="signature_corner",
                        choices=["tl", "tr", "bl", "br"], default=None)

    # -------------------------------------------------------- bundle output
    parser.add_argument("--dither", dest="dither", action="store_true", default=None)
    parser.add_argument("--dither-levels", dest="dither_levels", type=int, default=None)
    parser.add_argument("--background-value", dest="background_value", type=float, default=None)

    polarity = parser.add_mutually_exclusive_group()
    polarity.add_argument("--black-is-deep", dest="black_is_deep", action="store_true")
    polarity.add_argument("--white-is-deep", dest="black_is_deep", action="store_false")
    parser.set_defaults(black_is_deep=None)

    parser.add_argument("--naming", choices=["overwrite", "timestamp", "counter"], default=None,
                        help="Output file naming policy")
    parser.add_argument("--keep-history", dest="keep_history", action="store_true", default=None,
                        help="Never overwrite previous exports")

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


def _resolve_output(
    input_path: Path,
    output: str | None,
    output_dir_default: str,
    *,
    bundle_name: str | None = None,
) -> tuple[Path, str]:
    """Map ``--output`` (and optional ``--name``) to ``(parent_dir, bundle_name)``."""
    parent = Path(output) if output else Path(output_dir_default)
    stem = bundle_name.strip() if bundle_name and bundle_name.strip() else input_path.stem
    return parent, stem


def _print_bundle(bundle) -> None:
    print(f"Saved LightBurn heightmap to {bundle.lightburn_png}")
    print(f"Saved 16-bit master to {bundle.master16_png}")
    if bundle.preview_png:
        print(f"Saved preview to {bundle.preview_png}")
    if bundle.ramp_png:
        print(f"Saved calibration ramp to {bundle.ramp_png}")
    if bundle.subject_mask_png:
        print(f"Saved subject mask to {bundle.subject_mask_png}")
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
        print(f"Burn estimate ({be['width_mm']:.0f} x {be['height_mm']:.0f} mm): "
              f"{be['total_pretty']} total over {len(be['passes'])} pass(es)")
    if bundle.qa_findings:
        print(f"QA: {len(bundle.qa_findings)} finding(s):")
        for f in bundle.qa_findings:
            print(f"  [{f['severity']}] {f['code']}: {f['message']}")
    print(f"Done in {bundle.elapsed_s:.2f}s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.make_ramp:
        ramp_path = Path(args.make_ramp)
        ramp_path.parent.mkdir(parents=True, exist_ok=True)
        create_calibration_ramp().save(ramp_path)
        print(f"Saved calibration ramp to {ramp_path}")
        return

    app_settings = load_settings()

    if args.save_profile:
        from mopa.profiles import scaffold_profile

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
        parser.error("input is required unless --make-ramp or --save-profile is used")

    has_heightmap = bool(args.external_heightmap_path)
    has_sculptok = bool(getattr(args, "use_sculptok", False))
    if not (has_heightmap or has_sculptok):
        parser.error("Specify a heightmap source: --heightmap <path> or --use-sculptok.")
    if has_heightmap and has_sculptok:
        parser.error("--heightmap and --use-sculptok are mutually exclusive.")

    inputs = [Path(p) for p in args.input]
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

    profile_data: Dict[str, Any] = {}
    if args.profile:
        profile_data = load_profile(args.profile)

    # Target-object preset (optional). Layers in BEFORE explicit CLI
    # overrides so user flags still win.
    target = None
    if args.target:
        from mopa.target_presets import load_target_preset

        target = load_target_preset(args.target)
        target_overrides = dict(target.heightmap_overrides)
        # Push polarity_invert into the heightmap settings block so
        # merge_profile_settings sees it like any other key.
        target_overrides.setdefault("polarity_invert", target.polarity_invert)
        # Apply target overrides as the *base* layer; CLI flags override them.
        cli_overrides = _heightmap_overrides(args)
        # CLI args default to None when not set — drop those so the target
        # preset's value survives.
        cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        # Use the profile + target as the base, then apply CLI on top.
        settings = merge_profile_settings(profile_data, target_overrides)
        settings = merge_profile_settings({"heightmap": settings}, cli_overrides)
        # Print-size defaults from the target unless the user supplied them.
        if args.print_width_mm is None:
            args.print_width_mm = target.print_width_mm
        if args.print_height_mm is None:
            args.print_height_mm = target.print_height_mm
    else:
        overrides = _heightmap_overrides(args)
        settings = merge_profile_settings(profile_data, overrides)

    # --use-sculptok auto-pull: configure the client up-front so we can
    # do a single balance check before burning any credits. Per-input
    # API calls happen inside the loop below.
    sculptok_client = None
    sculptok_params = None
    if has_sculptok:
        from mopa.settings import resolve_sculptok_api_key
        from mopa.sculptok_client import (
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

    service = HeightmapService(app_settings=app_settings)
    multi = len(inputs) > 1
    for index, input_path in enumerate(inputs):
        output_dir, stem = _resolve_output(
            input_path, args.output, app_settings.output.directory,
            bundle_name=args.bundle_name if not multi else None,
        )

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
            write_subject_mask=bool(args.write_subject_mask),
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
            from mopa.sculptok_client import SculptokInsufficientCreditsError

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
            request,
            profile_name=args.profile,
            profile_data=profile_data,
            input_path=input_path,
        )
        _print_bundle(bundle)


if __name__ == "__main__":
    main()
