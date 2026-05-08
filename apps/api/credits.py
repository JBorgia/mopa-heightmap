"""Per-user credit tracking via Supabase Postgres.

Uses the service-role key (bypasses RLS) so the API backend can read and
mutate any user's credit row without going through the user's JWT.

Environment variables required in production:
    SUPABASE_URL          e.g. https://xyzxyz.supabase.co
    SUPABASE_SERVICE_KEY  service_role key (never expose this to the frontend)

When either variable is absent, the module operates in dev mode:
    - get_credits() returns a large stub value (999)
    - deduct_credit() succeeds silently without touching any database
    - add_credits()  succeeds silently
"""
from __future__ import annotations

import os
from typing import Optional

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_DEV_MODE = not _SUPABASE_URL or not _SERVICE_KEY

_client_cache: Optional[object] = None


def _client():
    global _client_cache
    if _client_cache is None:
        try:
            from supabase import create_client  # type: ignore[import]
            _client_cache = create_client(_SUPABASE_URL, _SERVICE_KEY)
        except ImportError as exc:
            raise RuntimeError(
                "supabase-py is not installed. Add it via: pip install supabase"
            ) from exc
    return _client_cache


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_credits(user_id: str) -> int:
    """Return the user's current credit balance. Returns 999 in dev mode."""
    if _DEV_MODE:
        return 999
    result = (
        _client()
        .table("user_credits")
        .select("credits")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if result.data is None:
        # First-time user — free tier gets 3 lifetime credits on first access.
        _ensure_row(user_id)
        return 3
    return int(result.data["credits"])


def deduct_credit(user_id: str, amount: int = 1) -> int:
    """Atomically deduct *amount* credits.

    Returns the new balance. Raises ``InsufficientCreditsError`` if the user
    has fewer than *amount* credits remaining.
    """
    if _DEV_MODE:
        return 999
    result = (
        _client()
        .rpc("deduct_credit", {"p_user_id": user_id, "p_amount": amount})
        .execute()
    )
    new_balance = result.data
    if new_balance is None or new_balance < 0:
        raise InsufficientCreditsError(
            f"User {user_id} has insufficient credits (tried to deduct {amount})."
        )
    return int(new_balance)


def add_credits(user_id: str, amount: int) -> int:
    """Add *amount* credits to the user's balance. Returns new balance."""
    if _DEV_MODE:
        return 999
    result = (
        _client()
        .rpc("add_credits", {"p_user_id": user_id, "p_amount": amount})
        .execute()
    )
    return int(result.data)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ensure_row(user_id: str) -> None:
    """Insert a free-tier credit row if one doesn't exist yet."""
    _client().table("user_credits").upsert(
        {"user_id": user_id, "credits": 3, "tier": "free"},
        on_conflict="user_id",
        ignore_duplicates=True,
    ).execute()


class InsufficientCreditsError(Exception):
    """Raised by deduct_credit when the user has no credits left."""
