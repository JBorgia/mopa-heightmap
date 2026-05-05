"""Reactive pipeline runner with hash-based stage caching.

A :class:`Stage` is a pure(-ish) function with:

* a stable ``name`` (used as a cache namespace and dependency key),
* a list of upstream ``inputs`` (other stage names whose outputs are required),
* a ``params`` dict (any JSON-serialisable values affecting the output),
* a ``compute(deps, params)`` callable returning the stage output.

Each stage execution is keyed by a SHA-256 of:

* the stage name,
* a hash digest of every upstream output (recursive),
* a hash digest of the (sorted) params dict.

When the user tweaks a slider, the runner invalidates only the stages whose
key changes. Upstream stages (depth inference, normals, etc.) stay cached
until their *own* params change, so live preview re-runs in milliseconds.

The runner is intentionally synchronous and storage-agnostic — it does not
write to disk, mutate filesystem state, or import torch. Long-running stages
inject their own progress callbacks; the runner just orchestrates.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np


__all__ = [
    "Stage",
    "PipelineRunner",
    "StageResult",
    "DEFAULT_CACHE_CAPACITY",
    "HASH_DIGEST_LENGTH",
    "hash_payload",
]


# Number of stage results to keep in-memory before evicting in LRU order.
# Sized for the deepest expected pipeline (depth + normals + FC + mask +
# detail + ~12 color passes ≈ 20 stages) plus headroom for parameter sweeps.
DEFAULT_CACHE_CAPACITY: int = 64

# Truncated hash length used in cache keys. SHA-256 truncated to 24 hex chars
# (96 bits) gives ~10⁻¹⁴ collision probability across a 64-entry cache,
# which is well below any realistic interactive-session duration.
HASH_DIGEST_LENGTH: int = 24


# --------------------------------------------------------------------- hashing

def _hash_value(value: Any) -> str:
    """Return a deterministic short digest for any common Python/Numpy value."""
    h = hashlib.sha256()
    if isinstance(value, np.ndarray):
        h.update(b"ndarray")
        h.update(str(value.shape).encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (bytes, bytearray)):
        h.update(b"bytes")
        h.update(bytes(value))
    elif isinstance(value, str):
        h.update(b"str")
        h.update(value.encode("utf-8"))
    elif isinstance(value, (int, float, bool, type(None))):
        h.update(b"scalar")
        h.update(repr(value).encode("utf-8"))
    elif isinstance(value, Mapping):
        h.update(b"map")
        for k in sorted(value.keys(), key=lambda x: repr(x)):
            h.update(repr(k).encode("utf-8"))
            h.update(b"=")
            h.update(_hash_value(value[k]).encode("utf-8"))
    elif isinstance(value, (list, tuple)):
        h.update(b"seq")
        for item in value:
            h.update(_hash_value(item).encode("utf-8"))
            h.update(b",")
    else:
        # Fallback for anything else: rely on JSON repr if possible, else id
        # (which still gives a stable key within a single process).
        try:
            h.update(b"json")
            h.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            h.update(b"id")
            h.update(str(id(value)).encode("utf-8"))
    return h.hexdigest()[:HASH_DIGEST_LENGTH]


def hash_payload(*values: Any) -> str:
    """Public helper: deterministic short digest for a tuple of values."""
    h = hashlib.sha256()
    for v in values:
        h.update(_hash_value(v).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:HASH_DIGEST_LENGTH]


# --------------------------------------------------------------------- stage

@dataclass(frozen=True)
class Stage:
    """One node in the pipeline DAG."""

    name: str
    inputs: Sequence[str]
    params: Mapping[str, Any]
    compute: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    # Optional list of source-side hashes (e.g. the input image's hash) that
    # the stage author wants to mix into the cache key even though they
    # aren't an explicit upstream stage. Useful for the very first stage
    # whose only "input" is the raw image bytes.
    extra_keys: Sequence[Any] = field(default_factory=tuple)


@dataclass(frozen=True)
class StageResult:
    """Wrapper around a stage output, carrying its cache key for chaining."""

    name: str
    cache_key: str
    value: Any


# --------------------------------------------------------------------- runner

class PipelineRunner:
    """Synchronous DAG executor with LRU output cache.

    Stages are registered by name. ``run(target_stage_names)`` returns a dict
    mapping every requested name to its current value, computing only the
    stages whose cache key has changed.
    """

    def __init__(self, capacity: int = DEFAULT_CACHE_CAPACITY) -> None:
        self._stages: Dict[str, Stage] = {}
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._capacity = int(capacity)

    # --------------------------------------------------------- registration
    def add(self, stage: Stage) -> None:
        if stage.name in self._stages:
            raise ValueError(f"Stage already registered: {stage.name}")
        for dep in stage.inputs:
            if dep not in self._stages:
                raise KeyError(
                    f"Stage {stage.name!r} depends on unregistered stage {dep!r}"
                )
        self._stages[stage.name] = stage

    def replace(self, stage: Stage) -> None:
        """Re-register a stage (e.g. after a slider changed its params)."""
        if stage.name not in self._stages:
            raise KeyError(f"Stage not registered: {stage.name}")
        for dep in stage.inputs:
            if dep not in self._stages:
                raise KeyError(
                    f"Stage {stage.name!r} depends on unregistered stage {dep!r}"
                )
        self._stages[stage.name] = stage

    def names(self) -> List[str]:
        return list(self._stages.keys())

    # ----------------------------------------------------------------- exec
    def _key_for(self, name: str) -> str:
        stage = self._stages[name]
        dep_keys = [self._key_for(dep) for dep in stage.inputs]
        return hash_payload(stage.name, dep_keys, stage.params, list(stage.extra_keys))

    def run(self, targets: Sequence[str]) -> Dict[str, StageResult]:
        """Compute ``targets`` (and all transitive deps), reusing cached values."""
        order = self._topo_order(targets)
        results: Dict[str, StageResult] = {}
        for name in order:
            stage = self._stages[name]
            key = self._key_for(name)
            if key in self._cache:
                value = self._cache[key]
                self._cache.move_to_end(key)
            else:
                deps = {dep: results[dep].value for dep in stage.inputs}
                value = stage.compute(deps, stage.params)
                self._cache[key] = value
                while len(self._cache) > self._capacity:
                    self._cache.popitem(last=False)
            results[name] = StageResult(name=name, cache_key=key, value=value)
        return {n: results[n] for n in targets}

    def _topo_order(self, targets: Sequence[str]) -> List[str]:
        order: List[str] = []
        visited: set[str] = set()

        def visit(n: str, stack: tuple[str, ...]) -> None:
            if n in visited:
                return
            if n in stack:
                cycle = " -> ".join(stack[stack.index(n):] + (n,))
                raise ValueError(f"Cycle detected in pipeline: {cycle}")
            if n not in self._stages:
                raise KeyError(f"Target stage not registered: {n}")
            for dep in self._stages[n].inputs:
                visit(dep, stack + (n,))
            visited.add(n)
            order.append(n)

        for t in targets:
            visit(t, ())
        return order

    # ------------------------------------------------------------- inspect
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
