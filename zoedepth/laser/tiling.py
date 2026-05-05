from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size >= length:
        return [0]

    step = max(tile_size - overlap, 1)
    positions = list(range(0, max(length - tile_size, 0) + 1, step))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def infer_tiled_pil(
    model,
    image: Image.Image,
    tile_size: int = 1024,
    overlap: int = 128,
    pad_input: bool = True,
    with_flip_aug: bool = True,
) -> np.ndarray:
    width, height = image.size
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")

    x_positions = _tile_positions(width, tile_size, overlap)
    y_positions = _tile_positions(height, tile_size, overlap)

    acc = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for top in y_positions:
        for left in x_positions:
            crop = image.crop((left, top, min(left + tile_size, width), min(top + tile_size, height)))
            tile_depth = model.infer_pil(crop, pad_input=pad_input, with_flip_aug=with_flip_aug)
            tile_depth = np.asarray(tile_depth, dtype=np.float32)

            tile_h, tile_w = tile_depth.shape
            weight = np.ones((tile_h, tile_w), dtype=np.float32)
            acc[top:top + tile_h, left:left + tile_w] += tile_depth * weight
            weights[top:top + tile_h, left:left + tile_w] += weight

    weights = np.maximum(weights, 1e-6)
    return acc / weights