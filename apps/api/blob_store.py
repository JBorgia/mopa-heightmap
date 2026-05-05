"""Blob store: in-process store for temporary image blobs keyed by sha256.

Blobs are stored in a temp directory under ~/.mopa-heightmap/api-blobs/.
The store is process-scoped; it does not survive restarts.  Content-addressed:
writing the same bytes twice returns the same id.

Thread-safe for concurrent FastAPI worker threads.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

_BLOB_DIR = Path(os.path.expanduser("~")) / ".mopa-heightmap" / "api-blobs"
_BLOB_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_meta: Dict[str, Tuple[str, int]] = {}   # id -> (content_type, size_bytes)


def _id_for_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:40]


def store_bytes(data: bytes, content_type: str = "image/png") -> str:
    blob_id = _id_for_bytes(data)
    path = _BLOB_DIR / blob_id
    with _lock:
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        _meta[blob_id] = (content_type, len(data))
    return blob_id


def store_image(image: Image.Image, *, mode: str = "16bit") -> str:
    """Save a PIL Image to the blob store. Returns blob_id."""
    buf = io.BytesIO()
    if mode == "16bit":
        arr = np.asarray(image)
        if arr.dtype != np.uint16:
            arr = (np.clip(arr, 0, 1) * 65535).astype(np.uint16) if arr.max() <= 1.0 else arr.astype(np.uint16)
        Image.fromarray(arr).save(buf, format="PNG")
    else:
        image.save(buf, format="PNG")
    return store_bytes(buf.getvalue(), "image/png")


def store_heightmap(heightmap: np.ndarray) -> str:
    """Store a float32 [0,1] heightmap as a 16-bit PNG blob."""
    arr16 = (np.clip(heightmap, 0.0, 1.0) * 65535).astype(np.uint16)
    img = Image.fromarray(arr16, mode="I;16") if arr16.ndim == 2 else Image.fromarray(arr16)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return store_bytes(buf.getvalue(), "image/png")


def load_bytes(blob_id: str) -> Optional[bytes]:
    path = _BLOB_DIR / blob_id
    if not path.exists():
        return None
    return path.read_bytes()


def load_image(blob_id: str) -> Optional[Image.Image]:
    data = load_bytes(blob_id)
    if data is None:
        return None
    return Image.open(io.BytesIO(data))


def load_heightmap(blob_id: str) -> Optional[np.ndarray]:
    """Load a 16-bit PNG blob back to a float32 [0,1] array."""
    img = load_image(blob_id)
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32) / 255.0


def get_meta(blob_id: str) -> Optional[Tuple[str, int]]:
    with _lock:
        return _meta.get(blob_id)


def exists(blob_id: str) -> bool:
    return (_BLOB_DIR / blob_id).exists()
