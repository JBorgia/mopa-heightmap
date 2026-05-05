from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - runtime guard
    yaml = None


USER_PROFILES_ENV = "MOPA_HEIGHTMAP_PROFILES"
USER_PROFILES_DEFAULT = Path(os.path.expanduser("~")) / ".mopa-heightmap" / "profiles"


# (key, min, max) for numeric heightmap fields. None = unbounded.
_HEIGHTMAP_NUMERIC_RANGES: Dict[str, Tuple[float | None, float | None]] = {
    "near_percentile": (0.0, 50.0),
    "far_percentile": (50.0, 100.0),
    "gamma": (0.1, 5.0),
    "contrast": (0.1, 5.0),
    "midtone_boost": (-0.5, 0.5),
    "deep_limit": (0.0, 0.5),
    "surface_limit": (0.5, 1.0),
    "smooth_diameter": (1, 50),
    "smooth_strength": (0.0, 1.0),
    "sharpen": (0.0, 2.0),
    "sharpen_sigma": (0.1, 20.0),
    "background_threshold": (0.0, 1.0),
    "background_value": (0.0, 1.0),
    # Stage A — input conditioning numeric knobs.
    "input_clahe_clip": (0.5, 10.0),
    "input_clahe_grid": (2, 32),
    "input_denoise_strength": (0.1, 30.0),
    "input_specular_threshold": (128, 255),
    "input_max_dim": (0, 8192),
    # Stage C extras.
    "edge_refine_diameter": (1, 50),
    "edge_refine_sigma_color": (0.0, 1.0),
    "edge_refine_sigma_space": (0.1, 30.0),
    "dither_levels": (2, 1024),
    "target_depth_um": (0.0, 5000.0),
    "posterize_passes": (0, 4096),
    # Stage B detail injection.
    "detail_strength": (0.0, 1.0),
    "detail_highpass_radius": (1, 50),
    # Phase 2 — subject isolation.
    "subject_mask_feather_px": (0, 64),
    "subject_mask_threshold": (0.0, 1.0),
    # Phase 3 — relief composite.
    "relief_strength": (0.0, 1.0),
    "relief_pad_fraction": (0.0, 0.999),
    # Phase 3b — gradient-domain compression.
    "depth_unsharp_gamma": (0.05, 1.0),
    "depth_unsharp_blend": (0.0, 1.0),
    # Phase 4 — face-aware relief.
    "face_relief_strength": (0.0, 2.0),
    # Bilateral cross-filter.
    "depth_bilateral_diameter": (1, 50),
    "depth_bilateral_sigma_color": (0.0, 1.0),
    "depth_bilateral_sigma_space": (0.1, 50.0),
    # Signature pass.
    "signature_height_fraction": (0.005, 0.5),
    "signature_margin_fraction": (0.0, 0.5),
    "signature_depth_fraction": (0.0, 1.0),
    "pre_upscale_target_long_side": (64, 8192),
    "relief_stylize_steps": (4, 200),
    "relief_stylize_guidance": (0.0, 20.0),
    "relief_stylize_controlnet_strength": (0.0, 2.0),
    "relief_stylize_seed": (0, 2_147_483_647),
    "relief_stylize_blend": (0.0, 1.0),
}

_HEIGHTMAP_BOOL_KEYS = {
    "flatten_background",
    "input_white_balance",
    "input_clahe",
    "input_denoise",
    "input_remove_specular",
    "edge_refine",
    "dither",
    "detail_subject_mask",
    "detail_invert",
    "subject_mask_enabled",
    "relief_enabled",
    "depth_unsharp_enabled",
    "face_relief_enabled",
    "auto_orient_face",
    "delight_enabled",
    "depth_bilateral_enabled",
    "pre_upscale_enabled",
    "relief_stylize_enabled",
}
_HEIGHTMAP_STRING_KEYS = {
    "smooth",
    "detail_mode",
    "subject_mask_backend",
    "relief_normals_backend",
    "delight_backend",
    "signature_text",
    "signature_corner",
    "pre_upscale_resolver",
    "relief_stylize_backend",
}
_VALID_SMOOTH_VALUES = {"none", "off", "bilateral", "gaussian"}
_VALID_DETAIL_MODES = {"off", "luminance", "highpass", "both"}
# Backend keys are validated only as "non-empty string"; the registry owns
# the authoritative list and rejects unknown keys at load time.
_FREE_FORM_STRING_KEYS = {"subject_mask_backend", "relief_normals_backend"}

_KNOWN_TOP_LEVEL_KEYS = {
    "name",
    "machine",
    "lightburn_mode",
    "black_is_deep",
    "heightmap",
    "lightburn_starting_point",
    "color_recipes",
    "calibration_lut",
    "__profile_path__",
}


class ProfileValidationError(ValueError):
    """Raised when a YAML profile fails schema validation."""

    def __init__(self, profile_path: str, errors: List[str]) -> None:
        self.profile_path = profile_path
        self.errors = errors
        joined = "\n  - ".join(errors)
        super().__init__(f"Profile validation failed for {profile_path}:\n  - {joined}")


def get_user_profiles_dir() -> Path:
    """Per-user override directory.

    Honors $MOPA_HEIGHTMAP_PROFILES, otherwise ~/.mopa-heightmap/profiles/.
    The directory is *not* created here; callers handle absence.
    """
    override = os.environ.get(USER_PROFILES_ENV)
    if override:
        return Path(override).expanduser()
    return USER_PROFILES_DEFAULT


def get_builtin_profiles_dirs() -> List[Path]:
    """Candidate directories shipped with the install.

    1. Repo-relative `<repo>/profiles/` (editable installs and source checkouts).
    2. Wheel asset path `mopa_heightmap_assets/profiles` (pip-installed wheels;
       see pyproject.toml force-include).
    """
    candidates: List[Path] = []
    repo_dir = Path(__file__).resolve().parents[2] / "profiles"
    candidates.append(repo_dir)

    # Wheel asset directory lives next to the top-level packages.
    try:
        import zoedepth  # noqa: F401
        site_root = Path(zoedepth.__file__).resolve().parent.parent
        candidates.append(site_root / "mopa_heightmap_assets" / "profiles")
    except Exception:  # pragma: no cover
        pass

    return candidates


def get_profiles_dir() -> Path:
    """Primary profiles directory used for the legacy 'where do they live?' question.

    Prefers the user-scope dir if it exists, otherwise the first builtin dir
    that exists, otherwise the canonical user-scope path (which may not exist
    yet). New code should prefer `iter_profile_dirs()` instead.
    """
    user = get_user_profiles_dir()
    if user.exists():
        return user
    for d in get_builtin_profiles_dirs():
        if d.exists():
            return d
    return user


def iter_profile_dirs() -> List[Path]:
    """All directories searched for profiles, in priority order."""
    dirs: List[Path] = []
    user = get_user_profiles_dir()
    if user.exists():
        dirs.append(user)
    for d in get_builtin_profiles_dirs():
        if d.exists() and d not in dirs:
            dirs.append(d)
    return dirs


def resolve_profile_path(profile_name_or_path: str) -> Path:
    candidate = Path(profile_name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate

    for base_dir in iter_profile_dirs():
        for suffix in ("", ".yaml", ".yml"):
            resolved = base_dir / f"{profile_name_or_path}{suffix}"
            if resolved.exists():
                return resolved

    raise FileNotFoundError(f"Profile not found: {profile_name_or_path}")


def list_profiles() -> List[str]:
    """All profile names, user-scope first, deduped, machine_* hidden."""
    seen: List[str] = []
    for base in iter_profile_dirs():
        for child in sorted(base.iterdir()):
            if child.suffix.lower() not in {".yaml", ".yml"}:
                continue
            if child.name.startswith("machine_"):
                continue
            if child.stem not in seen:
                seen.append(child.stem)
    return seen


def validate_profile(data: Dict[str, Any], profile_path: str = "<memory>") -> None:
    errors: List[str] = []

    unknown_top = set(data.keys()) - _KNOWN_TOP_LEVEL_KEYS
    if unknown_top:
        errors.append(f"unknown top-level keys: {sorted(unknown_top)}")

    if "black_is_deep" in data and not isinstance(data["black_is_deep"], bool):
        errors.append("black_is_deep must be a boolean")

    heightmap = data.get("heightmap", {})
    if not isinstance(heightmap, dict):
        errors.append("heightmap must be a mapping")
        heightmap = {}

    for key, value in heightmap.items():
        if key in _HEIGHTMAP_NUMERIC_RANGES:
            low, high = _HEIGHTMAP_NUMERIC_RANGES[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"heightmap.{key} must be a number")
                continue
            if low is not None and value < low:
                errors.append(f"heightmap.{key}={value} is below minimum {low}")
            if high is not None and value > high:
                errors.append(f"heightmap.{key}={value} is above maximum {high}")
        elif key in _HEIGHTMAP_BOOL_KEYS:
            if not isinstance(value, bool):
                errors.append(f"heightmap.{key} must be a boolean")
        elif key in _HEIGHTMAP_STRING_KEYS:
            if not isinstance(value, str):
                errors.append(f"heightmap.{key} must be a string")
            elif key == "signature_text":
                # Empty string is meaningful — disables the signature
                # pass — so we don't reject it here.
                pass
            elif not value.strip():
                errors.append(f"heightmap.{key} must be a non-empty string")
            elif key == "smooth" and value.lower() not in _VALID_SMOOTH_VALUES:
                errors.append(
                    f"heightmap.{key}={value!r} must be one of {sorted(_VALID_SMOOTH_VALUES)}"
                )
            elif key == "detail_mode" and value.lower() not in _VALID_DETAIL_MODES:
                errors.append(
                    f"heightmap.{key}={value!r} must be one of {sorted(_VALID_DETAIL_MODES)}"
                )
            elif key == "signature_corner" and value.lower() not in {"tl", "tr", "bl", "br"}:
                errors.append(
                    f"heightmap.{key}={value!r} must be one of ['bl', 'br', 'tl', 'tr']"
                )
            # _FREE_FORM_STRING_KEYS pass through after the non-empty check;
            # the runtime registry rejects unknown backend names with a
            # descriptive error, so we don't duplicate that list here.
        else:
            errors.append(f"heightmap.{key} is not a recognized parameter")

    near = heightmap.get("near_percentile")
    far = heightmap.get("far_percentile")
    if isinstance(near, (int, float)) and isinstance(far, (int, float)) and far <= near:
        errors.append(
            f"heightmap.far_percentile ({far}) must be greater than near_percentile ({near})"
        )

    deep = heightmap.get("deep_limit")
    surf = heightmap.get("surface_limit")
    if isinstance(deep, (int, float)) and isinstance(surf, (int, float)) and surf < deep:
        errors.append(
            f"heightmap.surface_limit ({surf}) must be >= deep_limit ({deep})"
        )

    # Optional calibration LUT block.
    if "calibration_lut" in data:
        lut_raw = data["calibration_lut"]
        if not isinstance(lut_raw, dict):
            errors.append("calibration_lut must be a mapping")
        else:
            samples = lut_raw.get("samples")
            if not isinstance(samples, list) or len(samples) < 2:
                errors.append("calibration_lut.samples must be a list with >= 2 entries")
            else:
                for i, entry in enumerate(samples):
                    if isinstance(entry, (list, tuple)) and len(entry) == 2 \
                            and all(isinstance(v, (int, float)) for v in entry):
                        continue
                    if isinstance(entry, dict) and "gray" in entry and "depth_um" in entry:
                        continue
                    errors.append(f"calibration_lut.samples[{i}] is malformed")

    # Optional color recipes block.
    if "color_recipes" in data:
        try:
            from .color_recipes import recipes_from_profile  # local import
            recipes_from_profile(data)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))

    if errors:
        raise ProfileValidationError(profile_path, errors)


def load_profile(profile_name_or_path: str, *, validate: bool = True) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load material profiles. Install pyyaml or use the provided environment file."
        )

    path = resolve_profile_path(profile_name_or_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ProfileValidationError(str(path), ["top-level YAML must be a mapping"])

    if validate:
        validate_profile(data, str(path))

    data["__profile_path__"] = str(path)
    return data


# ---------------------------------------------------------- profile authoring
_DEFAULT_LIGHTBURN_BLOCK: Dict[str, Any] = {
    "speed": 2000,
    "passes": 50,
    "line_interval": 0.04,
    "power": 80,
    "frequency": 200,
    "pulse_width": 100,
    "angle_increment": 0,
}


def scaffold_profile(
    name: str,
    heightmap_settings: Dict[str, Any],
    *,
    machine: str = "JPT MOPA fiber",
    lightburn_mode: str = "3D Sliced",
    black_is_deep: bool = True,
    lightburn_starting_point: Dict[str, Any] | None = None,
    target_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a starter YAML profile with the given settings.

    By default writes to the user-scope dir so the new profile shows up first
    in `list_profiles()`. Returns the absolute path of the created file.
    Raises FileExistsError unless `overwrite=True`.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to write profiles. Install pyyaml first."
        )
    safe_name = name.strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or safe_name.startswith(".")
        or any(c in safe_name for c in r'\/:*?"<>|')
    ):
        raise ValueError(f"Invalid profile name: {name!r}")

    target = (target_dir or get_user_profiles_dir()).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{safe_name}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Profile already exists: {path}")

    # Strip values that match the engine default so the YAML stays focused on
    # what the operator actually customized.
    from .service import DEFAULT_SETTINGS as _ENGINE_DEFAULTS

    heightmap_block: Dict[str, Any] = {}
    for key, value in heightmap_settings.items():
        if key == "black_is_deep":
            continue  # promoted to top-level
        if key in _ENGINE_DEFAULTS and value == _ENGINE_DEFAULTS[key]:
            continue
        heightmap_block[key] = value

    payload: Dict[str, Any] = {
        "name": safe_name,
        "machine": machine,
        "lightburn_mode": lightburn_mode,
        "black_is_deep": bool(black_is_deep),
        "heightmap": heightmap_block,
        "lightburn_starting_point": dict(lightburn_starting_point or _DEFAULT_LIGHTBURN_BLOCK),
    }

    # Validate before writing so we never produce a broken profile on disk.
    validate_profile(payload, profile_path=str(path))

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path


def write_lut_to_profile(
    profile_name_or_path: str,
    lut_dict: Dict[str, Any],
    *,
    create_missing: bool = False,
) -> Path:
    """Persist a calibration_lut block into an existing user-scope profile.

    The profile YAML is loaded, the `calibration_lut` block is replaced (or
    added), and the file is written back atomically. The full file is
    re-validated before write so a bad LUT can't poison a working profile.

    If `create_missing` is True and the profile doesn't exist, a fresh user
    profile is scaffolded with engine defaults plus the given LUT.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required to write profiles.")

    try:
        path = resolve_profile_path(profile_name_or_path)
    except FileNotFoundError:
        if not create_missing:
            raise
        from .service import DEFAULT_SETTINGS as _ENGINE_DEFAULTS
        scaffold_profile(profile_name_or_path, dict(_ENGINE_DEFAULTS))
        path = resolve_profile_path(profile_name_or_path)

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ProfileValidationError(str(path), ["top-level YAML must be a mapping"])

    # Strip our internal marker if present.
    data.pop("__profile_path__", None)
    data["calibration_lut"] = dict(lut_dict)
    validate_profile(data, profile_path=str(path))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)
    tmp.replace(path)
    return path