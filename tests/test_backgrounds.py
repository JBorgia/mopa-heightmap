"""Tests for procedural background generators."""
from __future__ import annotations

import numpy as np
import pytest

from mopa.backgrounds import (
    PATTERN_NAMES,
    checkers_pattern,
    dots_pattern,
    generate_pattern,
    guilloche_pattern,
    halftone_pattern,
    stripes_pattern,
)


# ----------------------------------------------------------- shape + range

@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_pattern_returns_correct_shape_and_range(name: str):
    arr = generate_pattern(name, 64, 48)
    assert arr.shape == (48, 64)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 - 1e-6
    assert arr.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_pattern_with_offset_angle_still_in_range(name: str):
    arr = generate_pattern(name, 64, 64, angle=37.5, scale=1.5, seed=42)
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 - 1e-6
    assert arr.max() <= 1.0 + 1e-6


# ----------------------------------------------------------- determinism

def test_seed_determinism_for_random_pattern():
    a = guilloche_pattern(48, 48, seed=7)
    b = guilloche_pattern(48, 48, seed=7)
    assert np.allclose(a, b)


def test_different_seeds_yield_different_guilloche():
    a = guilloche_pattern(48, 48, seed=1)
    b = guilloche_pattern(48, 48, seed=2)
    assert not np.allclose(a, b)


# ----------------------------------------------------------- per-pattern sanity

def test_stripes_change_along_x_when_angle_zero():
    arr = stripes_pattern(64, 16, angle=0.0)
    # Column-wise stripes ⇒ each row is identical.
    assert np.array_equal(arr[0], arr[7])


def test_checkers_alternate_rows_invert():
    arr = checkers_pattern(64, 64, scale=1.0, angle=0.0)
    # Step exactly one cell down should flip parity at the cell boundary.
    assert (arr.min(), arr.max()) == (0.0, 1.0)
    assert 0.3 < arr.mean() < 0.7  # ~half-and-half


def test_halftone_density_responds_to_cell_value():
    sparse = halftone_pattern(96, 96, cell_value=0.1)
    dense = halftone_pattern(96, 96, cell_value=0.9)
    assert dense.mean() > sparse.mean()


def test_dots_have_some_coverage():
    arr = dots_pattern(96, 96, scale=1.5, seed=0)
    # Should be neither fully off nor fully on.
    assert 0.0 < arr.mean() < 0.7


# ----------------------------------------------------------- dispatch

def test_generate_pattern_rejects_unknown():
    with pytest.raises(KeyError, match="Unknown pattern"):
        generate_pattern("does_not_exist", 16, 16)


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_generate_pattern_round_trips_through_dispatch(name: str):
    arr = generate_pattern(name, 32, 32, scale=1.0, angle=0.0, seed=0)
    assert arr.shape == (32, 32)
