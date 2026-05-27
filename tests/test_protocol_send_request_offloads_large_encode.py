"""Pin: ``DqliteProtocol._send_request`` offloads large-request
encode to a worker thread via ``asyncio.to_thread`` so the
multi-MiB BLOB / TEXT param memcpy chain does not freeze the
event loop.

The prior shape called ``await self._send(self._encoder.encode(
request))`` at every single-request site in ``protocol.py``
(16 sites converted; the two handshake sites at lines 393 and
470 deliberately keep direct ``_send`` because they prepend
8-byte handshake bytes and bundle a tiny ``ClientRequest`` into
one TCP segment by design). Every encode ran synchronously on
the loop thread before the first ``await`` in ``_send``'s
``drain()``. For an ``ExecRequest`` carrying a 64 MiB BLOB
param, the ~4 full-sized memcpys in ``encode_blob`` +
``encode_params_tuple`` consume hundreds of ms on x86 (multiple
seconds on Pi-class) before yielding.

The fix introduces a single ``_send_request(request)`` helper
that estimates encoded body size and threshold-gates the encode
via ``asyncio.to_thread`` when the projected size is at least
``_ENCODE_OFFLOAD_THRESHOLD`` (256 KiB — chosen to amortise the
~50 µs thread-hop cost while still catching all multi-MiB
shapes). Small messages (heartbeats, admin RPCs, fixed-shape
requests) stay in-loop to avoid the thread-hop overhead on the
hot path.

Mirrors the in-tree ``YamlNodeStore.set_nodes`` to_thread
discipline and aiohttp's compression-utils threshold-gated
``run_in_executor`` pattern (the only direct industry
precedent for size-gated encode offload).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import protocol as protocol_mod
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import requests as wire_requests

pytestmark = pytest.mark.asyncio


def _make_protocol_with_mock_writer() -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._writer = MagicMock()
    proto._writer.write = MagicMock()
    proto._writer.drain = AsyncMock()
    proto._timeout = 5.0
    # Real encoder so we exercise the actual encode path.
    from dqlitewire import MessageEncoder

    proto._encoder = MessageEncoder()
    proto._client_id = 1
    return proto


async def test_send_request_small_payload_stays_in_loop() -> None:
    """A heartbeat-class request (zero or tiny params) encodes
    in-loop — no ``asyncio.to_thread`` hop. Pins the threshold
    fast-path so heartbeats / admin RPCs do not pay the
    ~50 µs hop cost per call.
    """
    proto = _make_protocol_with_mock_writer()

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        # LeaderRequest is the canonical zero-param admin RPC.
        await proto._send_request(wire_requests.LeaderRequest())

    assert to_thread_calls == [], (
        f"small request encode unexpectedly offloaded: {to_thread_calls!r}. "
        f"Heartbeat-class requests must stay in-loop to avoid the "
        f"~50 µs thread-hop cost on every call."
    )
    # ``_writer.write`` is a MagicMock; mypy can't see the
    # call_count attribute through the typed Protocol stub.
    assert proto._writer.write.call_count == 1  # type: ignore[attr-defined]


async def test_send_request_large_blob_param_dispatches_via_to_thread() -> None:
    """An ``ExecRequest`` carrying a 1 MiB BLOB param (well above
    the 256 KiB gate) MUST be encoded on a worker thread so the
    loop is not frozen during the multi-pass memcpy.
    """
    proto = _make_protocol_with_mock_writer()

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    big_blob = b"\x00" * (1 << 20)  # 1 MiB
    request = wire_requests.ExecRequest(
        db_id=1,
        stmt_id=0,
        params=[big_blob],
    )

    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(request)

    assert len(to_thread_calls) == 1, (
        f"large-blob request must be offloaded to ``asyncio.to_thread``; "
        f"got {len(to_thread_calls)} hops."
    )
    # ``_writer.write`` is a MagicMock; mypy can't see the
    # call_count attribute through the typed Protocol stub.
    assert proto._writer.write.call_count == 1  # type: ignore[attr-defined]


async def test_send_request_threshold_boundary_offloads_at_threshold() -> None:
    """A payload exactly at the threshold offloads; below the
    threshold stays in-loop. Pins the boundary so future tuning
    is intentional.
    """
    proto = _make_protocol_with_mock_writer()
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD

    # Just BELOW the threshold — stays in-loop.
    small_blob = b"\x00" * (threshold - 1024)
    small_request = wire_requests.ExecRequest(
        db_id=1,
        stmt_id=0,
        params=[small_blob],
    )

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(small_request)
    assert to_thread_calls == [], (
        f"payload just below threshold should stay in-loop; saw offload calls: {to_thread_calls!r}"
    )

    # AT or above the threshold — offloads.
    big_blob = b"\x00" * (threshold + 1024)
    big_request = wire_requests.ExecRequest(
        db_id=1,
        stmt_id=0,
        params=[big_blob],
    )
    to_thread_calls.clear()
    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(big_request)
    assert len(to_thread_calls) == 1, (
        f"payload at/above threshold should offload; got {len(to_thread_calls)} hops."
    )


async def test_send_request_threshold_is_documented_constant() -> None:
    """The threshold lives as a module-level ``Final`` constant so
    operators can grep for it and so a future tuning lands at
    a single site.
    """
    assert hasattr(protocol_mod, "_ENCODE_OFFLOAD_THRESHOLD")
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    # Sanity: must be in the 64 KiB - 1 MiB band per the prior-art
    # convergence (aiohttp 4 KiB compression / asyncpg COPY 512 KiB
    # chunk / psycopg3 128 KiB buffer).
    assert 64 * 1024 <= threshold <= 1024 * 1024, (
        f"_ENCODE_OFFLOAD_THRESHOLD={threshold} is outside the sensible 64 KiB - 1 MiB band"
    )
