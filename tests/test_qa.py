"""Tests for :mod:`zoedepth.laser.qa`."""
from __future__ import annotations

import numpy as np
import pytest

from zoedepth.laser.qa import (
    DEFAULT_BG_FLOATER_STD,
    DEFAULT_MIN_DYNAMIC_RANGE,
    DEFAULT_MIN_SUBJECT_COVERAGE,
    QAFinding,
    check_background_floater,
    check_dynamic_range,
    check_floating_hair,
    check_mirror_asymmetry,
    check_specular_pits,
    check_subject_coverage,
    qa_report,
)


def _clean_heightmap(h: int = 64, w: int = 64) -> np.ndarray:
    """Subject in the centre, perfectly flat background at 1.0."""
    out = np.full((h, w), 1.0, dtype=np.float32)
    out[16:48, 16:48] = np.linspace(0.0, 0.7, 32 * 32, dtype=np.float32).reshape(32, 32)
    return out


# ----------------------------------------------------------- background floater

def test_clean_background_passes():
    findings = check_background_floater(_clean_heightmap())
    assert findings == []


def test_noisy_background_triggers_warning():
    """A background with std > threshold but every pixel still inside the
    check's "is-background" filter (±0.02 of 1.0) must trip the floater."""
    hm = _clean_heightmap()
    bg = np.abs(hm - 1.0) < 0.005
    # Deterministic ±0.018 split → std = 0.018 > 0.015 threshold, and every
    # pixel stays within ±0.02 of 1.0 so it survives the check's mask.
    flat = bg.flatten()
    indices = np.flatnonzero(flat)
    odd, even = indices[::2], indices[1::2]
    flat_view = hm.reshape(-1)
    flat_view[odd] = 1.0 + 0.018
    flat_view[even] = 1.0 - 0.018
    findings = check_background_floater(hm)
    assert findings
    assert findings[0].code == "bg_floater"
    assert findings[0].severity == "warning"


# ----------------------------------------------------------- subject coverage

def test_normal_coverage_passes():
    findings = check_subject_coverage(_clean_heightmap())
    assert findings == []


def test_tiny_subject_warns():
    hm = np.full((64, 64), 1.0, dtype=np.float32)
    hm[31:33, 31:33] = 0.5  # 4 px of 4096 = 0.1 %
    findings = check_subject_coverage(hm)
    assert findings and findings[0].code == "subject_too_small"


def test_full_frame_subject_info():
    hm = np.full((64, 64), 0.5, dtype=np.float32)
    findings = check_subject_coverage(hm)
    assert findings and findings[0].code == "subject_fills_frame"
    assert findings[0].severity == "info"


# ----------------------------------------------------------- dynamic range

def test_full_dynamic_range_passes():
    hm = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    findings = check_dynamic_range(hm)
    assert findings == []


def test_compressed_range_warns():
    hm = np.full((32, 32), 0.5, dtype=np.float32)
    hm[8:24, 8:24] = 0.55     # 5% span
    findings = check_dynamic_range(hm)
    assert findings and findings[0].code == "low_dynamic_range"


def test_empty_heightmap_errors():
    hm = np.full((4, 4), np.nan, dtype=np.float32)
    findings = check_dynamic_range(hm)
    assert findings and findings[0].severity == "error"


# ----------------------------------------------------------- mirror asymmetry

def test_symmetric_subject_no_warning():
    hm = _clean_heightmap()
    findings = check_mirror_asymmetry(hm)
    assert findings == []


def test_asymmetric_subject_flagged():
    hm = np.full((64, 64), 1.0, dtype=np.float32)
    hm[16:48, 16:32] = 0.0   # left half only
    hm[16:48, 32:48] = 0.9   # right half much shallower
    findings = check_mirror_asymmetry(hm, threshold=0.05)
    assert findings and findings[0].code == "mirror_asymmetry"


# ----------------------------------------------------------- specular pits

def test_specular_pit_detected_when_photo_provided():
    hm = np.full((32, 32), 0.5, dtype=np.float32)
    photo = np.full((32, 32, 3), 250, dtype=np.uint8)
    # Make a deep pit that aligns with the bright region — both > 50 px.
    hm[8:24, 8:24] = 0.1
    findings = check_specular_pits(hm, photo)
    assert findings and findings[0].code == "specular_as_pit"


def test_specular_check_no_op_without_photo():
    hm = _clean_heightmap()
    assert check_specular_pits(hm, None) == []


# ----------------------------------------------------------- floating hair

def test_floating_hair_when_top_brighter():
    hm = np.full((64, 64), 1.0, dtype=np.float32)
    hm[:10, 16:48] = 0.95     # top strip subject, very shallow (close to surface)
    hm[10:48, 16:48] = 0.2    # body subject, deeper
    findings = check_floating_hair(hm)
    assert findings and findings[0].code == "floating_hair"


# ----------------------------------------------------------- aggregator

def test_qa_report_concatenates_findings():
    hm = _clean_heightmap()
    findings = qa_report(hm)
    # All findings must be QAFinding instances.
    for f in findings:
        assert isinstance(f, QAFinding)
        assert f.severity in {"info", "warning", "error"}


def test_qa_report_with_photo_runs_specular_check():
    hm = np.full((32, 32), 0.5, dtype=np.float32)
    photo = np.full((32, 32, 3), 250, dtype=np.uint8)
    hm[8:24, 8:24] = 0.1
    findings = qa_report(hm, photo=photo)
    codes = [f.code for f in findings]
    assert "specular_as_pit" in codes


# ----------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert DEFAULT_BG_FLOATER_STD == 0.015
    assert DEFAULT_MIN_DYNAMIC_RANGE == 0.15
    assert DEFAULT_MIN_SUBJECT_COVERAGE == 0.03
