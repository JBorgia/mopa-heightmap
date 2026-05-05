from __future__ import annotations

import numpy as np


def background_mask_from_threshold(heightmap: np.ndarray, threshold: float) -> np.ndarray:
    mapped = np.asarray(heightmap, dtype=np.float32)
    if mapped.ndim != 2:
        raise ValueError(f"Expected a 2D heightmap, received shape {mapped.shape!r}")
    return mapped >= threshold


def flatten_background_region(heightmap: np.ndarray, threshold: float, fill_value: float = 1.0) -> np.ndarray:
    mapped = np.asarray(heightmap, dtype=np.float32).copy()
    mapped[background_mask_from_threshold(mapped, threshold)] = np.clip(fill_value, 0.0, 1.0)
    return mapped