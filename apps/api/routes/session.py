"""WebSocket /ws/session/{id} — real-time state mirror for the SPA.

Clients connect and receive server-push events whenever backend state changes
(render complete, mask ready, etc.).  The payload mirrors the SignalTree slice
that changed so the SPA can update without polling.

Events pushed by server:
  - ``render.complete``  {heightmap_id, preview_id, elapsed_s, image_hash}
  - ``mask.complete``    {mask_id, coverage_pct}
  - ``progress``         {step, total, message}
  - ``error``            {code, message, hint?}
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["session"])

# session_id -> set of connected websockets
_sessions: Dict[str, Set[WebSocket]] = {}
_sessions_lock = asyncio.Lock()


@router.websocket("/session/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    async with _sessions_lock:
        _sessions.setdefault(session_id, set()).add(websocket)
    try:
        while True:
            # Keep connection alive; client pings with {"type":"ping"}.
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        async with _sessions_lock:
            _sessions.get(session_id, set()).discard(websocket)


async def broadcast(session_id: str, event: str, payload: dict) -> None:
    """Push an event to all clients connected to a session.  Fire-and-forget."""
    msg = json.dumps({"event": event, "payload": payload})
    sockets = _sessions.get(session_id, set()).copy()
    for ws in sockets:
        try:
            await ws.send_text(msg)
        except Exception:  # noqa: BLE001
            pass
