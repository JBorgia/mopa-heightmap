"""LAB k-means color quantization for MOPA color-pass assignment.

Goal: take the source photo, find the K most-prominent perceptual colors,
and emit one binary mask per cluster. Each mask becomes a separate
:class:`mopa.stages.EngravingPass` with the cut parameters of
the chosen :class:`MaterialProfile` row of the same name.

Why LAB k-means and not RGB:
    Perceptual color difference (ΔE) is dominated by L*a*b* distance, not
    RGB. K-means in LAB clusters along human-meaningful color axes; the
    same image quantised in RGB would routinely group blacks with reds
    when their luma is similar.

What "prominent" means here:
    A two-stage pick: histogram-bin in 32×32×32 LAB cells, then run
    k-means seeded with the densest K cells. This is much faster than
    sklearn k-means on a full-resolution image and tends to find the
    same answer (the dense cells *are* where the cluster centroids will
    converge). Pure NumPy — no sklearn dependency.

Default behaviour:
    The K masks are named ``"C00"``, ``"C01"``, …, ``"C{K-1}"`` so they
    map directly onto the standard MOPA card naming scheme. The user
    overrides the assignment in the wizard if they want a particular
    photo color to fire a particular card row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "quantize_to_color_masks",
    "ColorClusterResult",
    "DEFAULT_K",
    "DEFAULT_KMEANS_ITERATIONS",
    "DEFAULT_BIN_RESOLUTION",
    "DEFAULT_DOWNSAMPLE_LONG_SIDE",
]


# Number of clusters by default — matches a typical 6–8 colour MOPA chart.
DEFAULT_K: int = 6

# K-means refinement iterations. The histogram-seeded centroids settle in
# 4–8 iterations on natural images; 16 is a generous ceiling.
DEFAULT_KMEANS_ITERATIONS: int = 16

# LAB histogram bin resolution used for centroid seeding. 32^3 = 32 768
# bins; on a megapixel image each bin holds ~30 samples on average.
DEFAULT_BIN_RESOLUTION: int = 32

# Cap on the long side of the image actually fed to k-means. K-means runs
# in O(N · K · iter) so capping at ~256 px keeps quantisation < 100 ms
# on a megapixel input. The resulting cluster centroids are then applied
# back to the full-resolution image to compute the actual per-pixel masks.
DEFAULT_DOWNSAMPLE_LONG_SIDE: int = 256


@dataclass(frozen=True)
class ColorClusterResult:
    """K-means output: per-cluster centroid (LAB), assignment mask, count."""

    name: str                   # "C00".."C{K-1}"
    lab_centroid: np.ndarray    # (3,) float32 — for downstream UI swatches
    rgb_centroid: np.ndarray    # (3,) uint8 in 0..255 — same purpose
    pixel_count: int            # how many source pixels fell in this cluster
    mask: np.ndarray            # (H, W) float32 in {0, 1} at source resolution


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (..., 3) uint8 / float RGB array into CIE LAB float32.

    Uses the D65 white point and the standard CIE conversion path. We avoid
    a hard dependency on opencv's color-conversion routines — they're fast
    but not part of the cv2-headless install on every CI machine.
    """
    arr = rgb.astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0

    # sRGB -> linear RGB.
    threshold = 0.04045
    linear = np.where(
        arr <= threshold,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    )

    # Linear RGB -> XYZ (D65).
    rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ rgb_to_xyz.T

    # XYZ -> LAB (D65 reference white).
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    f = xyz / ref
    eps = (6.0 / 29.0) ** 3
    f = np.where(
        f > eps,
        np.cbrt(f),
        f * (1.0 / (3.0 * (6.0 / 29.0) ** 2)) + (4.0 / 29.0),
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_rgb_to_lab` — for swatch previews only."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    f_inv = lambda f: np.where(f > delta, f ** 3, 3 * delta ** 2 * (f - 4.0 / 29.0))
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz = np.stack([f_inv(fx) * ref[0], f_inv(fy) * ref[1], f_inv(fz) * ref[2]], axis=-1)

    xyz_to_rgb = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    rgb_lin = xyz @ xyz_to_rgb.T
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    rgb = np.where(
        rgb_lin <= 0.0031308,
        rgb_lin * 12.92,
        1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
    )
    return np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _seed_from_histogram(
    pixels_lab: np.ndarray,
    k: int,
    *,
    bin_res: int = DEFAULT_BIN_RESOLUTION,
) -> np.ndarray:
    """Pick K seed centroids from the densest LAB histogram cells.

    LAB ranges: L in [0, 100], a/b roughly in [-128, 127]. We min-max into
    bin space, vote into a 3-D histogram, and pick the K-largest bins.
    """
    L = pixels_lab[:, 0]
    a = pixels_lab[:, 1]
    b = pixels_lab[:, 2]
    L_bin = np.clip((L / 100.0 * bin_res).astype(np.int32), 0, bin_res - 1)
    a_bin = np.clip(((a + 128.0) / 256.0 * bin_res).astype(np.int32), 0, bin_res - 1)
    b_bin = np.clip(((b + 128.0) / 256.0 * bin_res).astype(np.int32), 0, bin_res - 1)
    flat = L_bin * bin_res * bin_res + a_bin * bin_res + b_bin
    counts = np.bincount(flat, minlength=bin_res ** 3)
    top = np.argsort(counts)[-k:][::-1]

    seeds = np.empty((k, 3), dtype=np.float32)
    for i, code in enumerate(top):
        bL = code // (bin_res * bin_res)
        ba = (code // bin_res) % bin_res
        bb = code % bin_res
        seeds[i, 0] = (bL + 0.5) / bin_res * 100.0
        seeds[i, 1] = (ba + 0.5) / bin_res * 256.0 - 128.0
        seeds[i, 2] = (bb + 0.5) / bin_res * 256.0 - 128.0
    return seeds


def _kmeans(
    pixels_lab: np.ndarray,
    seeds: np.ndarray,
    *,
    iterations: int = DEFAULT_KMEANS_ITERATIONS,
) -> np.ndarray:
    """Standard k-means; returns the converged centroids ``(K, 3)``."""
    centroids = seeds.astype(np.float32, copy=True)
    n_pixels = pixels_lab.shape[0]
    if n_pixels == 0:
        return centroids
    for _ in range(int(iterations)):
        # Squared L2 in LAB ≈ ΔE_76 squared. Good enough for centroid
        # convergence; we don't need ΔE_2000 here.
        d2 = (
            (pixels_lab[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(axis=-1)
        labels = np.argmin(d2, axis=1)
        new_centroids = np.empty_like(centroids)
        moved = False
        for k in range(centroids.shape[0]):
            members = pixels_lab[labels == k]
            if members.shape[0] == 0:
                new_centroids[k] = centroids[k]
                continue
            new_centroids[k] = members.mean(axis=0)
            if not np.allclose(new_centroids[k], centroids[k], atol=1e-3):
                moved = True
        centroids = new_centroids
        if not moved:
            break
    return centroids


def quantize_to_color_masks(
    image: Image.Image,
    *,
    k: int = DEFAULT_K,
    subject_mask: Optional[np.ndarray] = None,
    long_side_for_kmeans: int = DEFAULT_DOWNSAMPLE_LONG_SIDE,
    name_prefix: str = "C",
) -> List[ColorClusterResult]:
    """Cluster ``image`` into K perceptual color groups.

    Parameters
    ----------
    image
        Source PIL image (RGB).
    k
        Number of clusters / output masks. Practical range: 2–10. The
        MOPA color cards themselves carry ~50–100 entries; ``k`` here is
        how many *distinct color regions* the user wants to engrave from
        this photo, not the size of the card.
    subject_mask
        Optional ``(H, W)`` float32 mask in ``[0, 1]``. Only pixels with
        ``subject_mask >= 0.5`` participate in clustering, and the final
        per-cluster masks are zeroed outside the subject. Lets us keep
        the BiRefNet-flattened background out of the color planner.
    long_side_for_kmeans
        K-means runs on a downscaled copy of the image for speed; the
        cluster *assignment* is then applied to every pixel of the full
        image. Set to 0 to skip the downscale.
    name_prefix
        Cluster naming convention. ``"C"`` produces ``"C00"``, ``"C01"``,
        etc. — matching the LightBurn card row names.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2; got {k}")
    rgb_full = np.asarray(image.convert("RGB"), dtype=np.uint8)
    H, W = rgb_full.shape[:2]

    long_side = max(H, W)
    if long_side_for_kmeans and long_side > long_side_for_kmeans:
        scale = long_side_for_kmeans / float(long_side)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        small = image.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        rgb_small = np.asarray(small, dtype=np.uint8)
    else:
        rgb_small = rgb_full

    lab_small = _rgb_to_lab(rgb_small).reshape(-1, 3)

    if subject_mask is not None:
        mask_small = _resample_mask(subject_mask, rgb_small.shape[:2])
        keep = mask_small.reshape(-1) >= 0.5
        cluster_pixels = lab_small[keep] if keep.any() else lab_small
    else:
        cluster_pixels = lab_small

    seeds = _seed_from_histogram(cluster_pixels, k)
    centroids = _kmeans(cluster_pixels, seeds)

    lab_full = _rgb_to_lab(rgb_full).reshape(-1, 3)
    d2 = ((lab_full[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    labels = np.argmin(d2, axis=1).reshape(H, W)

    final_subject = (
        _resample_mask(subject_mask, (H, W)) >= 0.5
        if subject_mask is not None
        else np.ones((H, W), dtype=bool)
    )

    rgb_centroids = _lab_to_rgb_uint8(centroids)

    out: List[ColorClusterResult] = []
    for i in range(centroids.shape[0]):
        mask = ((labels == i) & final_subject).astype(np.float32)
        out.append(ColorClusterResult(
            name=f"{name_prefix}{i:02d}",
            lab_centroid=centroids[i].astype(np.float32),
            rgb_centroid=rgb_centroids[i],
            pixel_count=int(mask.sum()),
            mask=mask,
        ))
    # Sort by descending population so the most-prominent color is C00.
    out.sort(key=lambda r: r.pixel_count, reverse=True)
    # Re-name after the sort.
    return [
        ColorClusterResult(
            name=f"{name_prefix}{i:02d}",
            lab_centroid=r.lab_centroid,
            rgb_centroid=r.rgb_centroid,
            pixel_count=r.pixel_count,
            mask=r.mask,
        )
        for i, r in enumerate(out)
    ]


def _resample_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resample a 2-D mask to ``target_shape``."""
    if mask.shape == target_shape:
        return mask.astype(np.float32, copy=False)
    pil = Image.fromarray(np.clip(mask, 0.0, 1.0).astype(np.float32), mode="F")
    pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def color_masks_for_planner(
    clusters: Iterable[ColorClusterResult],
) -> Dict[str, np.ndarray]:
    """Adapt a list of :class:`ColorClusterResult` to the planner's contract."""
    return {c.name: c.mask for c in clusters}
