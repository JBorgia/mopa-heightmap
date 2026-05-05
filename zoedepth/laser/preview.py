from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _to_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def _make_histogram(heightmap: np.ndarray, width: int = 512, height: int = 160) -> Image.Image:
    histogram, _ = np.histogram(heightmap.flatten(), bins=64, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32)
    if histogram.max() > 0:
        histogram /= histogram.max()

    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    bar_width = max(width // len(histogram), 1)
    for index, value in enumerate(histogram):
        left = index * bar_width
        right = min(width, left + bar_width - 1)
        top = int((1.0 - value) * (height - 12))
        draw.rectangle((left, top, right, height - 1), fill=(220, 220, 220))
    return canvas


def create_shaded_preview(heightmap: np.ndarray) -> Image.Image:
    mapped = np.clip(np.asarray(heightmap, dtype=np.float32), 0.0, 1.0)
    grad_y, grad_x = np.gradient(mapped)
    normals = np.dstack((-grad_x, -grad_y, np.ones_like(mapped)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)

    light = np.array([0.35, -0.4, 0.85], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = np.clip((normals * light).sum(axis=2), 0.0, 1.0)
    gray = np.repeat((mapped * 255.0)[..., None], 3, axis=2)
    shaded = 0.45 * gray + 0.55 * (intensity[..., None] * 255.0)
    return _to_rgb(shaded)


def render_preview(heightmap: np.ndarray) -> Image.Image:
    mapped = np.clip(np.asarray(heightmap, dtype=np.float32), 0.0, 1.0)
    grayscale = Image.fromarray(np.round(mapped * 255.0).astype(np.uint8), mode="L").convert("RGB")
    shaded = create_shaded_preview(mapped)
    histogram = _make_histogram(mapped)

    tile_width = max(grayscale.width, shaded.width)
    panel_width = tile_width * 2
    panel_height = max(grayscale.height, shaded.height) + histogram.height + 32
    panel = Image.new("RGB", (panel_width, panel_height), (24, 24, 24))

    panel.paste(grayscale.resize((tile_width, grayscale.height)), (0, 0))
    panel.paste(shaded.resize((tile_width, shaded.height)), (tile_width, 0))
    panel.paste(histogram.resize((panel_width, histogram.height)), (0, panel_height - histogram.height))

    draw = ImageDraw.Draw(panel)
    draw.text((12, 12), "LightBurn grayscale", fill=(255, 255, 255))
    draw.text((tile_width + 12, 12), "Shaded relief preview", fill=(255, 255, 255))
    draw.text((12, panel_height - histogram.height - 24), "Histogram", fill=(255, 255, 255))
    return panel


def create_calibration_ramp(
    width: int = 2200,
    height: int = 256,
    levels: Sequence[int] | None = None,
) -> Image.Image:
    ramp_levels = list(levels or [255, 230, 204, 179, 153, 128, 102, 77, 51, 26, 0])
    canvas = np.zeros((height, width), dtype=np.uint8)
    band_width = max(width // len(ramp_levels), 1)
    for index, value in enumerate(ramp_levels):
        start = index * band_width
        end = width if index == len(ramp_levels) - 1 else min(width, (index + 1) * band_width)
        canvas[:, start:end] = int(np.clip(value, 0, 255))
    return Image.fromarray(canvas, mode="L")


def save_preview(image: Image.Image, output_path: str | Path) -> Path:
    target = Path(output_path)
    image.save(target)
    return target