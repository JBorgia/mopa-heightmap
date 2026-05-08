"""JWT authentication helpers for the MOPA FastAPI backend.

Uses supabase-py's auth.get_user() to validate bearer tokens — no JWT
secret needed in the environment. When SUPABASE_URL / SUPABASE_SERVICE_KEY
are absent (local dev without Supabase), all checks are bypassed.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_DEV_MODE = not _SUPABASE_URL or not _SERVICE_KEY

_client_cache: Optional[object] = None


def _client():
    global _client_cache
    if _client_cache is None:
        from supabase import create_client  # type: ignore[import]
        _client_cache = create_client(_SUPABASE_URL, _SERVICE_KEY)
    return _client_cache


def _verify_token(token: str) -> dict:
    """Call Supabase Auth to validate the user JWT. Returns the user dict."""
    try:
        resp = _client().auth.get_user(token)
        if resp is None or resp.user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        u = resp.user
        return {"sub": str(u.id), "email": u.email, "role": "authenticated"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Dependency helpers
# --------------------------------------------------------------------------- #

async def get_optional_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[dict]:
    """Return the user dict, or None if unauthenticated / Supabase not configured."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    if _DEV_MODE:
        return None  # dev mode — treat as anonymous
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return _verify_token(token)
    except HTTPException:
        return None


async def require_auth(
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """Raise 401 if the request carries no valid Supabase JWT."""
    if _DEV_MODE:
        return {"sub": "dev-user", "email": "dev@localhost", "role": "authenticated"}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required.")
    token = authorization.removeprefix("Bearer ").strip()
    return _verify_token(token)
