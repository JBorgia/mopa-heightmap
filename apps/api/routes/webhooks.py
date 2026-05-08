"""Polar.sh webhook handler — grants credits on subscription and order events.

Set these environment variables in your deployment:
    POLAR_WEBHOOK_SECRET   — from Polar dashboard → Webhooks → signing secret

Polar sends a SHA-256 HMAC signature in the ``webhook-signature`` header as
``sha256=<hex>``.  We verify it before touching the database.

Product IDs must match what you create in the Polar dashboard.  Map them to
credit quantities in CREDIT_GRANTS below.

Each Polar product should include ``{supabase_user_id: "<uid>"}`` in its
metadata (set this in the checkout redirect URL using ?metadata[...] or via
the Polar API) so we know which Supabase user to credit.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from ..credits import add_credits

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

_POLAR_SECRET = os.environ.get("POLAR_WEBHOOK_SECRET", "")

# Map Polar product IDs → credits to grant.
# Update these IDs after creating your products in the Polar dashboard.
CREDIT_GRANTS: dict[str, int] = {
    # Monthly subscriptions — credits per billing period
    "maker_monthly": 20,
    "shop_monthly": 75,
    # Annual subscriptions — full year of credits up front
    "maker_annual": 240,
    "shop_annual": 900,
    # One-off top-up pack
    "topup_10": 10,
}


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #

@router.post("/polar")
async def polar_webhook(
    request: Request,
    webhook_signature: str = Header(alias="webhook-signature", default=""),
) -> dict[str, str]:
    body = await request.body()
    _verify(body, webhook_signature)

    try:
        event: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    event_type: str = event.get("type", "")
    data: dict[str, Any] = event.get("data", {})

    if event_type in ("subscription.created", "subscription.updated"):
        _handle_subscription(data)
    elif event_type == "order.created":
        _handle_order(data)
    # Other event types are acknowledged but ignored.

    return {"received": "ok"}


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #

def _verify(body: bytes, signature_header: str) -> None:
    if not _POLAR_SECRET:
        return  # dev mode — skip verification
    expected = hmac.new(_POLAR_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(f"sha256={expected}", signature_header):
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")


def _extract_user_id(data: dict) -> str | None:
    return data.get("metadata", {}).get("supabase_user_id")


def _handle_subscription(data: dict) -> None:
    """Grant monthly/annual subscription credits."""
    user_id = _extract_user_id(data)
    if not user_id:
        return
    product_id: str = data.get("product_id", "")
    credits = CREDIT_GRANTS.get(product_id, 0)
    if credits > 0:
        add_credits(user_id, credits)


def _handle_order(data: dict) -> None:
    """Grant credits for one-off top-up purchases."""
    user_id = _extract_user_id(data)
    if not user_id:
        return
    product_id: str = data.get("product_id", "")
    credits = CREDIT_GRANTS.get(product_id, 0)
    if credits > 0:
        add_credits(user_id, credits)
