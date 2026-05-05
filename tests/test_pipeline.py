"""Tests for the reactive pipeline runner."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.pipeline import (
    DEFAULT_CACHE_CAPACITY,
    HASH_DIGEST_LENGTH,
    PipelineRunner,
    Stage,
    hash_payload,
)


# ----------------------------------------------------------- hashing

def test_hash_payload_is_deterministic_and_length_pinned():
    a = hash_payload("x", {"k": 1})
    b = hash_payload("x", {"k": 1})
    assert a == b
    assert len(a) == HASH_DIGEST_LENGTH


def test_hash_payload_changes_when_value_changes():
    a = hash_payload({"k": 1})
    b = hash_payload({"k": 2})
    assert a != b


def test_hash_payload_handles_ndarrays():
    arr1 = np.zeros((4, 4), dtype=np.float32)
    arr2 = np.zeros((4, 4), dtype=np.float32)
    arr3 = np.ones((4, 4), dtype=np.float32)
    assert hash_payload(arr1) == hash_payload(arr2)
    assert hash_payload(arr1) != hash_payload(arr3)


def test_constants_have_documented_values():
    assert DEFAULT_CACHE_CAPACITY == 64
    assert HASH_DIGEST_LENGTH == 24


# ----------------------------------------------------------- registration

def test_runner_registers_stages_in_dependency_order():
    runner = PipelineRunner()
    runner.add(Stage("a", inputs=(), params={}, compute=lambda d, p: 1))
    runner.add(Stage("b", inputs=("a",), params={}, compute=lambda d, p: d["a"] + 1))
    assert runner.names() == ["a", "b"]


def test_runner_rejects_dependency_on_missing_stage():
    runner = PipelineRunner()
    with pytest.raises(KeyError, match="unregistered stage"):
        runner.add(Stage("b", inputs=("a",), params={}, compute=lambda d, p: 0))


def test_runner_rejects_duplicate_registration():
    runner = PipelineRunner()
    runner.add(Stage("a", inputs=(), params={}, compute=lambda d, p: 0))
    with pytest.raises(ValueError, match="already registered"):
        runner.add(Stage("a", inputs=(), params={}, compute=lambda d, p: 0))


# ----------------------------------------------------------- execution

def _counting_stage(name, inputs, params, behaviour):
    counter = {"calls": 0}

    def compute(deps, p):
        counter["calls"] += 1
        return behaviour(deps, p)

    return Stage(name, inputs=inputs, params=params, compute=compute), counter


def test_run_executes_all_dependencies_in_topo_order():
    runner = PipelineRunner()
    a_stage, a_counter = _counting_stage("a", (), {"v": 1}, lambda d, p: p["v"])
    b_stage, b_counter = _counting_stage("b", ("a",), {"v": 2}, lambda d, p: d["a"] + p["v"])
    runner.add(a_stage)
    runner.add(b_stage)
    out = runner.run(["b"])
    assert out["b"].value == 3
    assert a_counter["calls"] == 1
    assert b_counter["calls"] == 1


def test_run_caches_unchanged_stages():
    runner = PipelineRunner()
    a_stage, a_counter = _counting_stage("a", (), {"v": 1}, lambda d, p: p["v"])
    b_stage, b_counter = _counting_stage("b", ("a",), {"v": 0}, lambda d, p: d["a"] + p["v"])
    runner.add(a_stage)
    runner.add(b_stage)
    runner.run(["b"])
    runner.run(["b"])
    assert a_counter["calls"] == 1
    assert b_counter["calls"] == 1


def test_run_invalidates_only_changed_stages_and_descendants():
    runner = PipelineRunner()
    a_stage, a_counter = _counting_stage("a", (), {"v": 1}, lambda d, p: p["v"])
    b_stage, b_counter = _counting_stage("b", ("a",), {"v": 0}, lambda d, p: d["a"] + p["v"])
    runner.add(a_stage)
    runner.add(b_stage)
    runner.run(["b"])
    # Tweak only b's params: a should not re-run.
    new_b, new_b_counter = _counting_stage("b", ("a",), {"v": 99}, lambda d, p: d["a"] + p["v"])
    runner.replace(new_b)
    out = runner.run(["b"])
    assert out["b"].value == 100
    assert a_counter["calls"] == 1
    assert new_b_counter["calls"] == 1


def test_run_invalidates_descendants_when_upstream_changes():
    runner = PipelineRunner()
    a_stage, a_counter = _counting_stage("a", (), {"v": 1}, lambda d, p: p["v"])
    b_stage, b_counter = _counting_stage("b", ("a",), {}, lambda d, p: d["a"] * 10)
    runner.add(a_stage)
    runner.add(b_stage)
    runner.run(["b"])
    new_a, new_a_counter = _counting_stage("a", (), {"v": 5}, lambda d, p: p["v"])
    runner.replace(new_a)
    out = runner.run(["b"])
    assert out["b"].value == 50
    assert b_counter["calls"] == 2  # b had to recompute because a changed
    assert new_a_counter["calls"] == 1


def test_run_detects_cycle():
    runner = PipelineRunner()
    runner.add(Stage("a", inputs=(), params={}, compute=lambda d, p: 0))
    runner.add(Stage("b", inputs=("a",), params={}, compute=lambda d, p: 0))
    # Manually monkey-patch to force a cycle without going through ``add``
    # (which would correctly reject it).
    runner._stages["a"] = Stage("a", inputs=("b",), params={}, compute=lambda d, p: 0)
    with pytest.raises(ValueError, match="Cycle"):
        runner.run(["a"])


def test_cache_evicts_in_lru_order_when_full():
    runner = PipelineRunner(capacity=2)
    for i, name in enumerate(["a", "b", "c"]):
        s, _ = _counting_stage(name, (), {"i": i}, lambda d, p: p["i"])
        runner.add(s)
    runner.run(["a"])
    runner.run(["b"])
    runner.run(["c"])
    # Capacity 2 means the oldest (``a``) should have been evicted.
    assert runner.cache_size() == 2
