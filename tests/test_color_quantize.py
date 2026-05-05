"""Tests for :mod:`zoedepth.laser.color_quantize`."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoedepth.laser.color_quantize import (
    DEFAULT_BIN_RESOLUTION,
    DEFAULT_DOWNSAMPLE_LONG_SIDE,
    DEFAULT_K,
    DEFAULT_KMEANS_ITERATIONS,
    ColorClusterResult,
    color_masks_for_planner,
    quantize_to_color_masks,
)


def _two_color_image(w: int = 64, h: int = 64) -> Image.Image:
    """Half red / half blue square."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (220, 30, 30)   # red
    arr[:, w // 2 :] = (30, 30, 220)   # blue
    return Image.fromarray(arr, "RGB")


def _three_color_image(w: int = 96, h: int = 32) -> Image.Image:
    """Vertical RGB stripes."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    third = w // 3
    arr[:, :third] = (220, 30, 30)         # red
    arr[:, third : 2 * third] = (30, 220, 30)  # green
    arr[:, 2 * third :] = (30, 30, 220)        # blue
    return Image.fromarray(arr, "RGB")


# ----------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert DEFAULT_K == 6
    assert DEFAULT_KMEANS_ITERATIONS == 16
    assert DEFAULT_BIN_RESOLUTION == 32
    assert DEFAULT_DOWNSAMPLE_LONG_SIDE == 256


# ----------------------------------------------------------- shape contracts

def test_quantize_returns_k_clusters_with_unique_names():
    image = _three_color_image()
    clusters = quantize_to_color_masks(image, k=3)
    assert len(clusters) == 3
    names = {c.name for c in clusters}
    assert names == {"C00", "C01", "C02"}


def test_quantize_masks_are_disjoint_and_sum_to_total_pixels():
    image = _three_color_image()
    clusters = quantize_to_color_masks(image, k=3)
    total = sum(c.pixel_count for c in clusters)
    expected = image.size[0] * image.size[1]
    assert total == expected


def test_quantize_mask_dtype_and_shape():
    image = _two_color_image(48, 48)
    clusters = quantize_to_color_masks(image, k=2)
    for c in clusters:
        assert isinstance(c, ColorClusterResult)
        assert c.mask.shape == (48, 48)
        assert c.mask.dtype == np.float32
        assert set(np.unique(c.mask).tolist()) <= {0.0, 1.0}


# ----------------------------------------------------------- semantics

def test_quantize_separates_distinct_colors():
    """Half red / half blue: cluster centroids should land on each side."""
    image = _two_color_image(32, 32)
    clusters = quantize_to_color_masks(image, k=2)
    # Largest cluster has ≥ 40 % of pixels (sanity — both halves are equal).
    biggest = max(c.pixel_count for c in clusters)
    assert biggest >= 0.4 * 32 * 32
    # The two centroids must land on opposite poles of the a* / b* axes.
    a0 = clusters[0].lab_centroid[1]
    a1 = clusters[1].lab_centroid[1]
    # Red vs. blue: red is positive a*, blue is near zero / slightly negative.
    assert (a0 > 0 and a1 <= a0) or (a1 > 0 and a0 <= a1)


def test_quantize_sorted_by_descending_population():
    image = _three_color_image(96, 32)
    clusters = quantize_to_color_masks(image, k=3)
    counts = [c.pixel_count for c in clusters]
    assert counts == sorted(counts, reverse=True)


# ----------------------------------------------------------- subject mask

def test_subject_mask_zeros_out_background():
    image = _two_color_image(32, 32)
    # Subject mask: only the right half (blue) is subject.
    subj = np.zeros((32, 32), dtype=np.float32)
    subj[:, 16:] = 1.0
    clusters = quantize_to_color_masks(image, k=2, subject_mask=subj)
    # The combined cluster masks must not light up any background pixels.
    union = np.zeros((32, 32), dtype=np.float32)
    for c in clusters:
        union = np.maximum(union, c.mask)
    assert union[:, :16].sum() == 0.0


# ----------------------------------------------------------- API contract

def test_color_masks_for_planner_keyed_by_name():
    image = _three_color_image()
    clusters = quantize_to_color_masks(image, k=3)
    masks = color_masks_for_planner(clusters)
    assert set(masks.keys()) == {c.name for c in clusters}
    for name, mask in masks.items():
        assert mask.dtype == np.float32


def test_quantize_rejects_k_below_two():
    image = _two_color_image()
    with pytest.raises(ValueError, match=">= 2"):
        quantize_to_color_masks(image, k=1)
