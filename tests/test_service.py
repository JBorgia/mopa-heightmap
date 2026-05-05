import numpy as np
from PIL import Image

from zoedepth.laser.service import (
    DEFAULT_SETTINGS,
    ExportRequest,
    HeightmapService,
    InferenceConfig,
    merge_profile_settings,
)
from zoedepth.laser.settings import AppSettings


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def infer_pil(self, image, pad_input=True, with_flip_aug=False):
        self.calls += 1
        w, h = image.size
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        return np.tile(x, (h, 1))


def _fake_loader_factory():
    fake = _FakeModel()

    def loader(model_name, device):
        return fake, device

    return fake, loader


def test_merge_profile_settings_overrides_take_precedence():
    profile = {"black_is_deep": False, "heightmap": {"gamma": 0.9}}
    s = merge_profile_settings(profile, {"gamma": 0.5, "smooth": None})
    assert s["gamma"] == 0.5
    assert s["black_is_deep"] is False
    # None overrides do not clobber profile defaults.
    assert s["smooth"] == DEFAULT_SETTINGS["smooth"]


def test_service_caches_depth_across_renders():
    fake, loader = _fake_loader_factory()
    svc = HeightmapService(app_settings=AppSettings(), model_loader=loader)
    img = Image.new("RGB", (32, 32), color=(120, 120, 120))
    cfg = InferenceConfig(model_name="ZoeD_NK", device="cpu")

    settings_a = merge_profile_settings(None, None)
    settings_b = dict(settings_a, gamma=0.5)

    svc.render(img, settings_a, cfg)
    svc.render(img, settings_b, cfg)
    assert fake.calls == 1  # second render reused cached depth


def test_service_export_writes_full_bundle(tmp_path):
    fake, loader = _fake_loader_factory()
    svc = HeightmapService(app_settings=AppSettings(), model_loader=loader)
    img = Image.new("RGB", (48, 48), color=(80, 90, 100))
    cfg = InferenceConfig(model_name="ZoeD_NK", device="cpu")
    settings = merge_profile_settings(None, None)
    request = ExportRequest(output_dir=tmp_path, base_stem="thing", write_preview=True, write_calibration_ramp=True)
    bundle = svc.export(img, settings, cfg, request)
    assert bundle.lightburn_png.exists()
    assert bundle.master16_png.exists()
    assert bundle.preview_png and bundle.preview_png.exists()
    assert bundle.ramp_png and bundle.ramp_png.exists()
    assert bundle.settings_json.exists()


def test_cache_key_includes_precision_and_resolution():
    """Different precision/resolution must NOT share a cached depth map."""
    fake, loader = _fake_loader_factory()
    svc = HeightmapService(app_settings=AppSettings(), model_loader=loader)
    img = Image.new("RGB", (32, 32), color=(120, 120, 120))
    settings = merge_profile_settings(None, None)

    cfg_a = InferenceConfig(model_name="ZoeD_NK", device="cpu", precision="fp32", inference_resolution=0)
    cfg_b = InferenceConfig(model_name="ZoeD_NK", device="cpu", precision="fp32", inference_resolution=256)
    cfg_c = InferenceConfig(model_name="ZoeD_NK", device="cpu", precision="fp16", inference_resolution=0)
    # CPU coerces fp16 -> fp32, so cfg_c should hit the cfg_a cache.
    svc.render(img, settings, cfg_a)
    svc.render(img, settings, cfg_b)
    svc.render(img, settings, cfg_c)
    assert fake.calls == 2  # cfg_a & cfg_b unique; cfg_c reuses cfg_a (fp16->fp32 on CPU)


def test_inference_resolution_downscales_input():
    """When inference_resolution caps the longest side, the model sees the smaller image."""
    seen_sizes: list[tuple[int, int]] = []

    class _RecordingFake:
        def infer_pil(self, image, pad_input=True, with_flip_aug=False):
            seen_sizes.append(image.size)
            import numpy as np
            w, h = image.size
            return np.zeros((h, w), dtype=np.float32)

    fake = _RecordingFake()

    def loader(model_name, device):
        return fake, device

    svc = HeightmapService(app_settings=AppSettings(), model_loader=loader)
    img = Image.new("RGB", (1024, 768), color=(100, 100, 100))
    cfg = InferenceConfig(model_name="ZoeD_NK", device="cpu", inference_resolution=384)
    depth, _ = svc.infer_depth(img, cfg)
    assert seen_sizes == [(384, 288)]
    # Depth must be resized back to the conditioned image's full size.
    assert depth.shape == (768, 1024)
