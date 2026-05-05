"""Tests for :mod:`zoedepth.laser.sculptok_client`.

All tests run against a fully-mocked ``requests.Session``; no network
calls. Every endpoint has a happy-path test plus an error-path test
covering the standard ``code != 0`` envelope failure.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zoedepth.laser.sculptok_client import (
    BASE_URL,
    PROMPT_PRICE_3D,
    PROMPT_PRICE_HD_FIX,
    PROMPT_PRICE_NORMAL,
    PROMPT_PRICE_PRO,
    PROMPT_PRICE_PRO_4K,
    PROMPT_PRICE_STL,
    STATUS_COMPLETED,
    STATUS_PROCESSING,
    SculptokAPIError,
    SculptokClient,
    SculptokDepthMapParams,
    SculptokHDFixParams,
    SculptokInsufficientCreditsError,
    SculptokSTLParams,
    SculptokThreeDParams,
)


# --------------------------------------------------------- helpers

def _envelope(data, code=0, msg="success"):
    """Produce a Sculptok-shaped response envelope."""
    return {"code": code, "msg": msg, "data": data}


def _mock_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    resp.content = b"PNG-binary-bytes"
    return resp


def _client_with_session(session):
    client = SculptokClient(api_key="test-key")
    client._session = session
    return client


# --------------------------------------------------------- constants

def test_constants_have_documented_values():
    assert BASE_URL == "https://api.sculptok.com/api-open"
    assert STATUS_PROCESSING == 1
    assert STATUS_COMPLETED == 2
    assert PROMPT_PRICE_NORMAL == 10
    assert PROMPT_PRICE_PRO == 15
    assert PROMPT_PRICE_PRO_4K == 30
    assert PROMPT_PRICE_HD_FIX == 2
    assert PROMPT_PRICE_3D == 10
    assert PROMPT_PRICE_STL == 3


# --------------------------------------------------------- DepthMapParams

def test_depth_params_defaults_target_pro_2k():
    p = SculptokDepthMapParams()
    assert p.style == "pro"
    assert p.version == "1.5"
    assert p.draw_hd == "2k"
    assert p.ext_info == "16bit"
    assert p.expected_cost() == PROMPT_PRICE_PRO


def test_depth_params_pro_4k_costs_30():
    p = SculptokDepthMapParams(draw_hd="4k")
    assert p.expected_cost() == PROMPT_PRICE_PRO_4K


def test_depth_params_normal_costs_10_regardless_of_hd():
    p = SculptokDepthMapParams(style="normal", draw_hd="4k")
    # draw_hd is ignored when style != pro.
    assert p.expected_cost() == PROMPT_PRICE_NORMAL


def test_depth_params_to_request_body_omits_pro_fields_when_normal():
    p = SculptokDepthMapParams(style="portrait")
    body = p.to_request_body("https://x")
    assert body["imageUrl"] == "https://x"
    assert body["style"] == "portrait"
    assert "version" not in body
    assert "draw_hd" not in body


def test_depth_params_to_request_body_includes_pro_fields_when_pro():
    p = SculptokDepthMapParams(style="pro", version="1.0", draw_hd="4k")
    body = p.to_request_body("https://x")
    assert body["version"] == "1.0"
    assert body["draw_hd"] == "4k"


def test_depth_params_rejects_exr_when_not_pro():
    with pytest.raises(ValueError, match="exr.*pro"):
        SculptokDepthMapParams(style="normal", ext_info="exr")


def test_depth_params_rejects_invalid_style():
    with pytest.raises(ValueError, match="style"):
        SculptokDepthMapParams(style="banana")


# --------------------------------------------------------- HDFixParams

def test_hd_fix_params_round_trip():
    p = SculptokHDFixParams(hd_fix=True, remove_back="general")
    assert p.expected_cost() == PROMPT_PRICE_HD_FIX
    body = p.to_request_body("https://x")
    assert body["hdFix"] == "true"
    assert body["removeBack"] == "general"


def test_hd_fix_params_omits_remove_back_when_none():
    p = SculptokHDFixParams(hd_fix=False, remove_back=None)
    body = p.to_request_body("https://x")
    assert body["hdFix"] == "false"
    assert "removeBack" not in body


def test_hd_fix_params_rejects_invalid_remove_back():
    with pytest.raises(ValueError, match="remove_back"):
        SculptokHDFixParams(remove_back="kawaii")


# --------------------------------------------------------- ThreeDParams

def test_3d_params_default_basic():
    p = SculptokThreeDParams()
    assert p.hd_fix == "basic"
    assert p.expected_cost() == PROMPT_PRICE_3D
    assert p.to_request_body("https://x")["hd_fix"] == "basic"


def test_3d_params_rejects_invalid_precision():
    with pytest.raises(ValueError, match="basic.*standard.*high"):
        SculptokThreeDParams(hd_fix="ultra")


# --------------------------------------------------------- STLParams

def test_stl_params_default_round_trip():
    p = SculptokSTLParams()
    body = p.to_request_body("https://x")
    assert body["image_url"] == "https://x"
    assert body["width_mm"] == 120.0
    assert body["min_thickness"] == 1.6
    assert body["max_thickness"] == 5.0
    assert body["invert"] is False
    assert body["scale_image"] == 50.0
    assert p.expected_cost() == PROMPT_PRICE_STL


def test_stl_params_rejects_thickness_inversion():
    with pytest.raises(ValueError, match="max_thickness must exceed"):
        SculptokSTLParams(min_thickness=5.0, max_thickness=2.0)


def test_stl_params_rejects_out_of_range_width():
    with pytest.raises(ValueError, match="width_mm"):
        SculptokSTLParams(width_mm=300.0)


# --------------------------------------------------------- credits

def test_get_credits_round_trip():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope({"point": 66}))
    c = _client_with_session(sess)
    assert c.get_credits() == 66
    sess.get.assert_called_once()
    args, kwargs = sess.get.call_args
    assert args[0] == f"{BASE_URL}/point/info"


def test_get_credits_history_returns_list():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope({
        "total": 1,
        "list": [{"id": "1", "remainValue": 50, "changeNum": -10, "remarks": "API Draw"}],
    }))
    c = _client_with_session(sess)
    history = c.get_credits_history(limit=5, page=1)
    assert len(history) == 1
    assert history[0]["remainValue"] == 50


def test_get_credits_unauthorized_surfaces_as_api_error():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope(None, code=401, msg="Unauthorized"))
    c = _client_with_session(sess)
    with pytest.raises(SculptokAPIError, match="Unauthorized"):
        c.get_credits()


def test_envelope_with_credit_message_raises_insufficient_credits():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope(None, code=402, msg="Not enough credits"))
    c = _client_with_session(sess)
    with pytest.raises(SculptokInsufficientCreditsError, match="credits"):
        c.get_credits()


# --------------------------------------------------------- upload

def test_upload_image_returns_src_url(tmp_path: Path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"FAKE-JPEG-BYTES")
    sess = MagicMock()
    sess.post.return_value = _mock_response(
        _envelope({"src": "https://mock/uploaded.png"})
    )
    c = _client_with_session(sess)
    url = c.upload_image(p)
    assert url == "https://mock/uploaded.png"
    args, kwargs = sess.post.call_args
    assert args[0] == f"{BASE_URL}/image/upload"
    assert "files" in kwargs
    assert kwargs["files"]["file"][0] == "photo.jpg"


def test_upload_image_missing_src_raises(tmp_path: Path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"x")
    sess = MagicMock()
    sess.post.return_value = _mock_response(_envelope({}))
    c = _client_with_session(sess)
    with pytest.raises(SculptokAPIError, match="no 'src'"):
        c.upload_image(p)


def test_upload_image_missing_file_raises(tmp_path: Path):
    c = _client_with_session(MagicMock())
    with pytest.raises(FileNotFoundError, match="not found"):
        c.upload_image(tmp_path / "does_not_exist.png")


# --------------------------------------------------------- submit + status

def test_submit_depth_map_returns_prompt_id():
    sess = MagicMock()
    sess.post.return_value = _mock_response(
        _envelope({"promptId": "abc123"})
    )
    c = _client_with_session(sess)
    pid = c.submit_depth_map("https://image.png")
    assert pid == "abc123"
    args, kwargs = sess.post.call_args
    assert args[0] == f"{BASE_URL}/draw/prompt"
    assert kwargs["json"]["imageUrl"] == "https://image.png"


def test_submit_hd_fix_uses_dedicated_endpoint():
    sess = MagicMock()
    sess.post.return_value = _mock_response(_envelope({"promptId": "hd1"}))
    c = _client_with_session(sess)
    pid = c.submit_hd_fix("https://image.png", SculptokHDFixParams(remove_back="general"))
    assert pid == "hd1"
    assert sess.post.call_args[0][0] == f"{BASE_URL}/draw/hd/prompt"


def test_submit_3d_draw_uses_dedicated_endpoint():
    sess = MagicMock()
    sess.post.return_value = _mock_response(_envelope({"promptId": "3d1"}))
    c = _client_with_session(sess)
    pid = c.submit_3d_draw("https://image.png")
    assert pid == "3d1"
    assert sess.post.call_args[0][0] == f"{BASE_URL}/draw/3d/prompt"


def test_submit_stl_uses_dedicated_endpoint():
    sess = MagicMock()
    sess.post.return_value = _mock_response(_envelope({"promptId": "stl1"}))
    c = _client_with_session(sess)
    pid = c.submit_image_to_stl("https://image.png")
    assert pid == "stl1"
    assert sess.post.call_args[0][0] == f"{BASE_URL}/draw/stl/prompt"


def test_get_drawing_status_decodes_completion():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope({
        "id": "1",
        "currentStep": 3,
        "status": STATUS_COMPLETED,
        "createDate": "2025-01-12 17:42:19",
        "userId": "u1",
        "upImageUrl": "https://up.png",
        "promptId": "pid1",
        "imgRecords": ["https://r0.png", "https://r1.png", "https://r2.png"],
    }))
    c = _client_with_session(sess)
    s = c.get_drawing_status("pid1")
    assert s.is_completed
    assert s.image_results[0] == "https://r0.png"
    assert s.current_step == 3


def test_get_drawing_status_decodes_in_progress():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope({
        "promptId": "pid1",
        "status": STATUS_PROCESSING,
        "currentStep": 1,
        "imgRecords": [],
    }))
    c = _client_with_session(sess)
    s = c.get_drawing_status("pid1")
    assert not s.is_completed
    assert not s.is_failed
    assert s.image_results == []


def test_get_drawing_history_returns_list():
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope({
        "total": 1,
        "list": [{"id": "1", "imgUrl": "https://h.png"}],
    }))
    c = _client_with_session(sess)
    history = c.get_drawing_history(limit=5, page=1)
    assert len(history) == 1
    assert history[0]["imgUrl"] == "https://h.png"


# --------------------------------------------------------- polling

def test_wait_for_completion_returns_final_status():
    sess = MagicMock()
    statuses = [
        _envelope({"promptId": "p1", "status": STATUS_PROCESSING, "imgRecords": []}),
        _envelope({"promptId": "p1", "status": STATUS_PROCESSING, "imgRecords": []}),
        _envelope({"promptId": "p1", "status": STATUS_COMPLETED, "imgRecords": ["https://r.png"]}),
    ]
    sess.get.side_effect = [_mock_response(s) for s in statuses]
    c = _client_with_session(sess)
    final = c.wait_for_completion("p1", interval_s=0.0, timeout_s=5.0)
    assert final.is_completed
    assert final.image_results == ["https://r.png"]
    assert sess.get.call_count == 3


def test_wait_for_completion_raises_on_failure_status():
    sess = MagicMock()
    sess.get.return_value = _mock_response(
        _envelope({"promptId": "p1", "status": 99, "imgRecords": []})
    )
    c = _client_with_session(sess)
    with pytest.raises(SculptokAPIError, match="failed"):
        c.wait_for_completion("p1", interval_s=0.0, timeout_s=5.0)


def test_wait_for_completion_times_out():
    sess = MagicMock()
    sess.get.return_value = _mock_response(
        _envelope({"promptId": "p1", "status": STATUS_PROCESSING, "imgRecords": []})
    )
    c = _client_with_session(sess)
    with pytest.raises(TimeoutError, match="not complete"):
        c.wait_for_completion("p1", interval_s=0.0, timeout_s=0.05)


# --------------------------------------------------------- generate_heightmap

def test_generate_heightmap_pre_flight_credit_check_blocks_when_low(tmp_path: Path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"x")
    sess = MagicMock()
    sess.get.return_value = _mock_response(_envelope({"point": 5}))
    c = _client_with_session(sess)
    with pytest.raises(SculptokInsufficientCreditsError, match="need 15"):
        c.generate_heightmap(p)


def test_generate_heightmap_full_path(tmp_path: Path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"x")
    out = tmp_path / "result.png"
    sess = MagicMock()
    sess.get.side_effect = [
        _mock_response(_envelope({"point": 100})),
        _mock_response(_envelope({"promptId": "p1", "status": STATUS_COMPLETED,
                                  "imgRecords": ["https://result.png"]})),
        _mock_response(_envelope({})),  # download
    ]
    sess.post.return_value = _mock_response(_envelope({"promptId": "p1", "src": "https://up.png"}))
    # Two POSTs: upload returns 'src', submit returns 'promptId'.
    sess.post.side_effect = [
        _mock_response(_envelope({"src": "https://up.png"})),
        _mock_response(_envelope({"promptId": "p1"})),
    ]
    c = _client_with_session(sess)

    result = c.generate_heightmap(p, out_path=out, poll_interval_s=0.0)
    assert result == out
    assert out.exists()
    # Sequence: get_credits (GET), upload (POST), submit (POST), poll (GET), download (GET).
    assert sess.get.call_count == 3
    assert sess.post.call_count == 2
