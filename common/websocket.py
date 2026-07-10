"""
Small helpers around FastAPI's WebSocket so server.py files stay thin.
Handles the "send many binary frames without blocking the event loop /
without racing a concurrent close" concerns that are easy to get wrong.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import orjson
from fastapi import WebSocket
from starlette.websockets import WebSocketState


async def send_json(ws: WebSocket, payload: Any) -> None:
    """Send a pydantic model or dict as compact JSON text."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if ws.application_state != WebSocketState.CONNECTED:
        return
    await ws.send_text(orjson.dumps(payload).decode())


async def send_binary(ws: WebSocket, data: bytes) -> None:
    if ws.application_state != WebSocketState.CONNECTED:
        return
    await ws.send_bytes(data)


async def stream_binary_frames(ws: WebSocket, frames: AsyncIterator[bytes]) -> int:
    """Send every frame from an async iterator; returns count sent, stops early on disconnect."""
    sent = 0
    async for frame in frames:
        if ws.application_state != WebSocketState.CONNECTED:
            break
        await ws.send_bytes(frame)
        sent += 1
    return sent


def is_connected(ws: WebSocket) -> bool:
    return ws.application_state == WebSocketState.CONNECTED
