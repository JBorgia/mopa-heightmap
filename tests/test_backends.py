"""Tests for the pluggable depth-inference backend registry.

We exercise:
  * the registry contents (DAv2 Small/Base default-on, Large opt-in only),
  * the ``DepthAnythingV2Wrapper`` sign-flip and shape contract using a
    minimal stub model + processor (no network, no GPU),
  * the service's loader dispatcher prefers the registry over hubconf.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from zoedepth.laser import backends as bk
from zoedepth.laser.backends import (
    BackendSpec,
    DepthAnythingV2Wrapper,
    get_backend,
    list_backends,
    register_backend,
)
from zoedepth.laser.service import _default_model_loader


# --------------------------------------------------------------------------- registry

def test_dav2_variants_registered_with_correct_licensing():
    small = get_backend("DAv2_Small")
    base = get_backend("DAv2_Base")
    large = get_backend("DAv2_Large")
    assert small is not None and not small.requires_opt_in
    assert base is not None and not base.requires_opt_in
    assert large is not None and large.requires_opt_in
    assert small.license == "Apache-2.0"
    assert base.license == "Apache-2.0"
    assert large.license.startswith("CC-BY-NC")


def test_list_backends_filters_opt_in():
    keys_default = {b.key for b in list_backends(include_opt_in=False)}
    keys_all = {b.key for b in list_backends(include_opt_in=True)}
    assert "DAv2_Large" in keys_all
    assert "DAv2_Large" not in keys_default
    assert {"DAv2_Small", "DAv2_Base"}.issubset(keys_default)


def test_register_backend_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="already registered"):
        register_backend(BackendSpec(
            key="DAv2_Base", label="dup", license="x",
            requires_opt_in=False, vram_estimate_mb=1, loader=lambda d: None,
        ))


# ---------------------------------------------------------------- DAv2 wrapper

class _StubProcessor:
    def __call__(self, images, return_tensors="pt"):
        # Resize-equivalent: just turn into a tiny tensor of the right shape.
        arr = np.asarray(images, dtype=np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)        # HWC -> CHW
        return {"pixel_values": torch.from_numpy(arr).unsqueeze(0)}


class _StubDAv2Model:
    """Stub HF DAv2 head: returns a controllable inverse-depth-like tensor."""

    def __init__(self, pattern: torch.Tensor):
        # pattern shape (H', W'); we'll repeat per-call.
        self._pattern = pattern
        self.calls = 0

    def __call__(self, pixel_values):
        self.calls += 1
        h, w = self._pattern.shape
        return SimpleNamespace(predicted_depth=self._pattern.unsqueeze(0).clone())


def test_dav2_wrapper_inverts_to_far_is_larger():
    # Inverse-depth-like: left column near (10), right column far (1).
    pattern = torch.tensor([[10.0, 5.0, 1.0]])  # 1x3
    model = _StubDAv2Model(pattern)
    wrapper = DepthAnythingV2Wrapper(model, _StubProcessor(), device="cpu")
    img = Image.new("RGB", (3, 1), color=(0, 0, 0))
    depth = wrapper.infer_pil(img, with_flip_aug=False)
    assert depth.shape == (1, 3)
    # Expect (10-10, 10-5, 10-1) = (0, 5, 9) → near now SMALL, far now LARGE.
    assert depth[0, 0] < depth[0, 1] < depth[0, 2]
    assert pytest.approx(depth[0, 0], abs=1e-5) == 0.0
    assert pytest.approx(depth[0, 2], abs=1e-5) == 9.0


def test_dav2_wrapper_flip_aug_averages_two_passes():
    pattern = torch.tensor([[10.0, 5.0, 1.0]])
    model = _StubDAv2Model(pattern)
    wrapper = DepthAnythingV2Wrapper(model, _StubProcessor(), device="cpu")
    img = Image.new("RGB", (3, 1), color=(0, 0, 0))
    _ = wrapper.infer_pil(img, with_flip_aug=True)
    assert model.calls == 2  # one for image, one for flipped image


def test_dav2_wrapper_resamples_to_input_size():
    # Model output is 2x2; input image is 6x4 — wrapper must upsample.
    pattern = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model = _StubDAv2Model(pattern)
    wrapper = DepthAnythingV2Wrapper(model, _StubProcessor(), device="cpu")
    img = Image.new("RGB", (6, 4), color=(0, 0, 0))   # PIL size = (W, H)
    depth = wrapper.infer_pil(img, with_flip_aug=False)
    assert depth.shape == (4, 6)


# -------------------------------------------------------- service dispatcher

def test_service_loader_prefers_registry_over_hubconf(monkeypatch):
    sentinel = object()

    def fake_load(key, device):
        assert key == "DAv2_Base"
        return sentinel, device

    monkeypatch.setattr(bk, "load_backend", fake_load)
    model, device = _default_model_loader("DAv2_Base", "cpu")
    assert model is sentinel
    assert device == "cpu"


def test_service_loader_falls_back_to_hubconf_for_unknown_registry_keys(monkeypatch):
    """ZoeD_* must continue to load through hubconf even after registry exists."""
    monkeypatch.setattr(bk, "get_backend", lambda key: None)

    captured = {}

    class _FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_ctor(pretrained=True):
        captured["pretrained"] = pretrained
        return _FakeModel()

    import hubconf
    monkeypatch.setattr(hubconf, "ZoeD_NK", fake_ctor, raising=False)
    model, device = _default_model_loader("ZoeD_NK", "cpu")
    assert isinstance(model, _FakeModel)
    assert captured == {"pretrained": True, "device": "cpu"}
    assert device == "cpu"
