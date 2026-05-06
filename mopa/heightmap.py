"""Heightmap I/O utilities — uint8/uint16 conversion, dither, file save.

Sculptok produces engraving-ready heightmaps; we don't reshape them. This
module is the small surface that turns a normalised float [0, 1] heightmap
into the bytes LightBurn reads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _as_float32_2d(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D heightmap array, received shape {data.shape!r}")
    return data


def floyd_steinberg_dither(heightmap: np.ndarray, levels: int = 256) -> np.ndarray:
    """Quantize a [0,1] heightmap to ``levels`` gray steps using Floyd-Steinberg.

    Returns a float32 array still in [0,1]. Caller chooses bit depth on save.
    """
    src = _as_float32_2d(heightmap).copy()
    levels = max(int(levels), 2)
    step = 1.0 / (levels - 1)
    h, w = src.shape
    out = np.zeros_like(src)
    for y in range(h):
        for x in range(w):
            old = src[y, x]
            new = round(old / step) * step
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                src[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    src[y + 1, x - 1] += err * (3 / 16)
                src[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    src[y + 1, x + 1] += err * (1 / 16)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_uint8(heightmap: np.ndarray) -> np.ndarray:
    mapped = np.clip(_as_float32_2d(heightmap), 0.0, 1.0)
    return np.round(mapped * 255.0).astype(np.uint8)


def to_uint16(heightmap: np.ndarray) -> np.ndarray:
    mapped = np.clip(_as_float32_2d(heightmap), 0.0, 1.0)
    return np.round(mapped * 65535.0).astype(np.uint16)


def save_heightmap_uint8(heightmap: np.ndarray, output_path: str | Path) -> Path:
    target = Path(output_path)
    Image.fromarray(to_uint8(heightmap), mode="L").save(target)
    return target


def save_heightmap_uint16(heightmap: np.ndarray, output_path: str | Path) -> Path:
    target = Path(output_path)
    Image.fromarray(to_uint16(heightmap), mode="I;16").save(target)
    return target
