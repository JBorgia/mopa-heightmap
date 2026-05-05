"""Tests for :mod:`zoedepth.laser.delighting`.

These tests exercise the registry and the wiring contract; the actual
Marigold pipeline is too heavy to load in CI, so we register a stub
delighter and verify the loader returns it correctly.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoedepth.laser import delighting as dl
from zoedepth.laser.delighting import (
    DEFAULT_DELIGHTER_KEY,
    DelighterSpec,
    get_delighter,
    list_delighters,
    load_delighter,
    register_delighter,
)


def test_default_delighter_registered_as_opt_in_nc():
    spec = get_delighter(DEFAULT_DELIGHTER_KEY)
    assert spec is not None
    assert spec.requires_opt_in is True
    assert spec.license.startswith("CC-BY-NC")


def test_list_delighters_filters_opt_in():
    keys_default = {d.key for d in list_delighters(include_opt_in=False)}
    keys_all = {d.key for d in list_delighters(include_opt_in=True)}
    # The default delighter is opt-in only, so it should be in `all` but
    # not in the `default` view.
    assert DEFAULT_DELIGHTER_KEY in keys_all
    assert DEFAULT_DELIGHTER_KEY not in keys_default


def test_register_duplicate_key_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_delighter(DelighterSpec(
            key=DEFAULT_DELIGHTER_KEY, label="dup", license="x",
            requires_opt_in=True, needs_gpu=True, vram_estimate_mb=0,
            loader=lambda d: None,
        ))


def test_load_unknown_key_raises():
    with pytest.raises(KeyError, match="No delighter"):
        load_delighter("not_a_real_one", "cpu")


# ----------------------------------------------------------- stub backend

class _StubDelighter:
    def albedo(self, image: Image.Image) -> Image.Image:
        # Return a constant-grey image of the same size.
        return Image.new("RGB", image.size, (128, 128, 128))


def test_register_and_load_custom_delighter():
    register_delighter(DelighterSpec(
        key="_test_stub_delighter", label="stub", license="MIT",
        requires_opt_in=False, needs_gpu=False, vram_estimate_mb=0,
        loader=lambda device: _StubDelighter(),
    ))
    inst, dev = load_delighter("_test_stub_delighter", "cpu")
    img = Image.new("RGB", (32, 32), (200, 50, 50))
    out = inst.albedo(img)
    assert dev == "cpu"
    assert out.size == img.size
    assert np.asarray(out)[16, 16, 0] == 128
