"""Pin: ``DqliteProtocol._send_request`` threshold-gates large encodes
onto ``asyncio.to_thread`` (at ``_ENCODE_OFFLOAD_THRESHOLD``) so a
multi-MiB BLOB/TEXT memcpy chain does not freeze the loop, while small
requests stay in-loop to avoid the thread-hop cost on the hot path.
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
    """A heartbeat-class request encodes in-loop, with no to_thread hop."""
    proto = _make_protocol_with_mock_writer()

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(wire_requests.LeaderRequest())

    assert to_thread_calls == [], (
        f"small request encode unexpectedly offloaded: {to_thread_calls!r}. "
        f"Heartbeat-class requests must stay in-loop to avoid the "
        f"~50 µs thread-hop cost on every call."
    )
    # mypy can't see call_count through the typed Protocol stub.
    assert proto._writer.write.call_count == 1  # type: ignore[attr-defined]


async def test_send_request_large_blob_param_dispatches_via_to_thread() -> None:
    """A 1 MiB BLOB param (above the gate) is encoded on a worker thread."""
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
    # mypy can't see call_count through the typed Protocol stub.
    assert proto._writer.write.call_count == 1  # type: ignore[attr-defined]


async def test_send_request_threshold_boundary_offloads_at_threshold() -> None:
    """At/above the threshold offloads; below stays in-loop."""
    proto = _make_protocol_with_mock_writer()
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD

    # Just below the threshold.
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

    # At/above the threshold.
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
    """The threshold is a module-level ``Final`` constant."""
    assert hasattr(protocol_mod, "_ENCODE_OFFLOAD_THRESHOLD")
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    assert 64 * 1024 <= threshold <= 1024 * 1024, (
        f"_ENCODE_OFFLOAD_THRESHOLD={threshold} is outside the sensible 64 KiB - 1 MiB band"
    )
