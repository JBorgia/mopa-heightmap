"""Tests for the super-resolution registry + auto-upscale helper."""
from __future__ import annotations

import pytest
from PIL import Image

from zoedepth.laser.super_resolution import (
    DEFAULT_RESOLVER_KEY,
    DEFAULT_TARGET_LONG_SIDE_PX,
    MAX_AUTO_UPSCALE_FACTOR,
    MIN_AUTO_UPSCALE_FACTOR,
    SuperResolverSpec,
    auto_upscale,
    get_resolver,
    list_resolvers,
    load_resolver,
    register_resolver,
)


def test_constants_have_documented_values():
    assert DEFAULT_RESOLVER_KEY == "lanczos"
    assert DEFAULT_TARGET_LONG_SIDE_PX == 1024
    assert MAX_AUTO_UPSCALE_FACTOR == 4.0
    assert MIN_AUTO_UPSCALE_FACTOR == 1.0


def test_default_backends_registered():
    keys = {s.key for s in list_resolvers()}
    assert {"lanczos", "bicubic", "realesrgan-x4plus"} <= keys


def test_default_resolvers_have_permissive_licences():
    """Real-ESRGAN x4 weights ship under BSD-3-Clause (commercial-OK), so
    every default resolver — including Real-ESRGAN — is non-opt-in."""
    assert get_resolver("lanczos").requires_opt_in is False
    assert get_resolver("bicubic").requires_opt_in is False
    assert get_resolver("realesrgan-x4plus").requires_opt_in is False


def test_list_resolvers_default_view_includes_realesrgan():
    keys = {s.key for s in list_resolvers(include_opt_in=False)}
    assert "lanczos" in keys
    assert "realesrgan-x4plus" in keys


def test_load_unknown_resolver_raises():
    with pytest.raises(KeyError):
        load_resolver("not-real", "cpu")


def test_register_duplicate_resolver_raises():
    spec = SuperResolverSpec(
        key="lanczos", label="x", license="MIT",
        requires_opt_in=False, max_scale=2.0, vram_estimate_mb=0,
        loader=lambda d: object(),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_resolver(spec)


def test_lanczos_upscale_changes_image_size():
    img = Image.new("RGB", (64, 32), color=(10, 20, 30))
    resolver, _ = load_resolver("lanczos", "cpu")
    out = resolver.upscale(img, 2.0)
    assert out.size == (128, 64)


def test_lanczos_upscale_noop_when_scale_le_one():
    img = Image.new("RGB", (64, 32))
    resolver, _ = load_resolver("lanczos", "cpu")
    assert resolver.upscale(img, 1.0).size == (64, 32)


def test_auto_upscale_skips_when_already_large_enough():
    img = Image.new("RGB", (DEFAULT_TARGET_LONG_SIDE_PX + 100, 100))
    out = auto_upscale(img)
    assert out.size == img.size


def test_auto_upscale_grows_small_image_to_target():
    img = Image.new("RGB", (200, 100))
    out = auto_upscale(img, target_long_side=400)
    assert max(out.size) == 400


def test_auto_upscale_caps_at_max_factor():
    img = Image.new("RGB", (100, 50))
    out = auto_upscale(img, target_long_side=10_000)
    # Capped at MAX_AUTO_UPSCALE_FACTOR == 4.0.
    assert max(out.size) == int(100 * MAX_AUTO_UPSCALE_FACTOR)


def test_auto_upscale_rejects_non_positive_target():
    img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError, match="positive"):
        auto_upscale(img, target_long_side=0)


def test_realesrgan_resolver_loadable_when_package_installed():
    """When ``realesrgan`` + ``basicsr`` are present, the loader returns a
    resolver with an ``upscale`` method ready to run inference."""
    realesrgan = pytest.importorskip("realesrgan")
    pytest.importorskip("basicsr.archs.rrdbnet_arch")
    resolver, dev = load_resolver("realesrgan-x4plus", "cpu")
    assert dev == "cpu"
    assert hasattr(resolver, "upscale")
