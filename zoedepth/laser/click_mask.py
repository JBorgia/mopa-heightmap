"""Click-driven subject mask refinement (SAM-2 style).

The default subject-mask backends (rembg/BiRefNet) produce one global
mask. Power users frequently want to *click* on the subject (or shift-
click on accessories) to add/subtract regions. This module exposes that
contract behind the same registry pattern as :mod:`subject_mask`.

Permissive default: ``flood-fill`` — pure NumPy connected-component grow
from each click, no weights, no GPU. Surprisingly good for high-contrast
subjects.

Opt-in heavy default: ``sam2-tiny`` (Apache-2.0 once Meta releases the
public weights; opt-in for now because the loader assumes
``segment_anything_2`` is installed).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from PIL import Image


__all__ = [
    "ClickMaskerSpec",
    "register_clicker",
    "get_clicker",
    "list_clickers",
    "load_clicker",
    "DEFAULT_CLICKER_KEY",
    "DEFAULT_FLOOD_TOLERANCE",
    "DEFAULT_FLOOD_MAX_FRACTION",
    "POINT_LABEL_POSITIVE",
    "POINT_LABEL_NEGATIVE",
]


# ----------------------------------------------------------- constants

DEFAULT_CLICKER_KEY: str = "flood-fill"

# Luminance distance (in [0, 1]) two pixels can differ by and still be
# considered the same region by the flood-fill clicker.
DEFAULT_FLOOD_TOLERANCE: float = 0.08

# Hard cap on flooded region size as a fraction of the image. Prevents a
# click on the background from filling the entire frame.
DEFAULT_FLOOD_MAX_FRACTION: float = 0.6

# Click labels mirror SAM-2's convention so the API survives a future
# backend swap.
POINT_LABEL_POSITIVE: int = 1
POINT_LABEL_NEGATIVE: int = 0


# ----------------------------------------------------------- registry

@dataclass(frozen=True)
class ClickMaskerSpec:
    """Metadata + loader for one click-driven mask backend."""

    key: str
    label: str
    license: str
    requires_opt_in: bool
    needs_gpu: bool
    vram_estimate_mb: int
    loader: Callable[[str], Any]      # device -> object with .infer(image, points, labels)


_REGISTRY: Dict[str, ClickMaskerSpec] = {}


def register_clicker(spec: ClickMaskerSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Click masker already registered: {spec.key}")
    _REGISTRY[spec.key] = spec


def get_clicker(key: str) -> ClickMaskerSpec | None:
    return _REGISTRY.get(key)


def list_clickers(include_opt_in: bool = True) -> Tuple[ClickMaskerSpec, ...]:
    items = sorted(_REGISTRY.values(), key=lambda s: s.key)
    if include_opt_in:
        return tuple(items)
    return tuple(s for s in items if not s.requires_opt_in)


def load_clicker(key: str, device: str) -> Tuple[Any, str]:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"No click masker registered for: {key!r}")
    return spec.loader(device), device


# ----------------------------------------------------------- default backends

class _FloodFillClicker:
    """BFS region grow over luminance with a tolerance threshold.

    Each positive-labelled point seeds a region; each negative-labelled
    point seeds an erase region. The output alpha is 1 where any positive
    region grew and not erased by a negative region.
    """

    def __init__(self, tolerance: float, max_fraction: float) -> None:
        self._tolerance = float(tolerance)
        self._max_fraction = float(max_fraction)

    def infer(
        self,
        image: Image.Image,
        points: Sequence[Tuple[int, int]],
        labels: Sequence[int],
    ) -> np.ndarray:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        h, w = gray.shape
        max_pixels = int(self._max_fraction * h * w)
        positive = np.zeros((h, w), dtype=bool)
        negative = np.zeros((h, w), dtype=bool)
        for (x, y), label in zip(points, labels):
            if not (0 <= x < w and 0 <= y < h):
                continue
            grown = self._flood(gray, (int(y), int(x)), max_pixels)
            if int(label) == POINT_LABEL_POSITIVE:
                positive |= grown
            else:
                negative |= grown
        alpha = np.where(positive & ~negative, 1.0, 0.0).astype(np.float32)
        return alpha

    def _flood(self, gray: np.ndarray, seed: Tuple[int, int], max_pixels: int) -> np.ndarray:
        h, w = gray.shape
        sy, sx = seed
        seed_value = gray[sy, sx]
        out = np.zeros((h, w), dtype=bool)
        out[sy, sx] = True
        queue: deque[Tuple[int, int]] = deque()
        queue.append((sy, sx))
        count = 1
        while queue and count < max_pixels:
            y, x = queue.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not out[ny, nx]:
                    if abs(gray[ny, nx] - seed_value) <= self._tolerance:
                        out[ny, nx] = True
                        queue.append((ny, nx))
                        count += 1
                        if count >= max_pixels:
                            break
        return out


class _SAM2Stub:
    """Loader-time guard for the SAM-2 opt-in backend."""

    def __init__(self, device: str) -> None:
        self._device = device
        try:
            import sam2  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SAM-2 is opt-in: pip install segment-anything-2 to enable "
                "the sam2-tiny click masker."
            ) from exc

    def infer(self, image, points, labels):
        raise RuntimeError(
            "SAM-2 inference not wired in this build; install the package "
            "and replace _SAM2Stub with the upstream predictor."
        )


def _make_flood_loader(tolerance: float, max_fraction: float) -> Callable[[str], Any]:
    def _load(_device: str) -> Any:
        return _FloodFillClicker(tolerance=tolerance, max_fraction=max_fraction)
    return _load


# ----------------------------------------------------------- registrations

register_clicker(ClickMaskerSpec(
    key="flood-fill",
    label="Flood-fill (CPU, instant)",
    license="MIT",
    requires_opt_in=False,
    needs_gpu=False,
    vram_estimate_mb=0,
    loader=_make_flood_loader(
        tolerance=DEFAULT_FLOOD_TOLERANCE,
        max_fraction=DEFAULT_FLOOD_MAX_FRACTION,
    ),
))


register_clicker(ClickMaskerSpec(
    key="sam2-tiny",
    label="SAM-2 Tiny (GPU, opt-in)",
    license="Apache-2.0",
    requires_opt_in=True,
    needs_gpu=True,
    vram_estimate_mb=1500,
    loader=lambda device: _SAM2Stub(device),
))
