"""Pin: ``DqliteProtocol._read_response`` offloads a multi-MiB wire decode to a worker
thread when the pending message exceeds ``_DECODE_OFFLOAD_THRESHOLD`` (256 KiB), so the
per-row/per-cell decode doesn't pin the loop and miss a heartbeat deadline."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire import MessageDecoder, MessageEncoder
from dqlitewire.constants import ValueType
from dqlitewire.messages.responses import EmptyResponse, FilesResponse, RowsResponse

pytestmark = pytest.mark.asyncio


def _build_rows_response_bytes(n_rows: int, n_cols: int) -> bytes:
    encoder = MessageEncoder()
    column_names = [f"col_{i}" for i in range(n_cols)]
    column_types = [ValueType.INTEGER] * n_cols
    rows = [[i + j for j in range(n_cols)] for i in range(n_rows)]
    row_types = [list(column_types) for _ in range(n_rows)]
    # int is a member of WireValue, so the rows arg is widening-safe.
    response = RowsResponse(
        column_names=column_names,
        column_types=column_types,
        row_types=row_types,
        rows=rows,  # type: ignore[arg-type]
        has_more=False,
    )
    return encoder.encode(response)


def _build_empty_response_bytes() -> bytes:
    encoder = MessageEncoder()
    return encoder.encode(EmptyResponse())


def _make_protocol_with_buffered_response(frame_bytes: bytes) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._writer = MagicMock()
    proto._writer.write = MagicMock()
    proto._writer.drain = AsyncMock()
    proto._timeout = 5.0
    proto._read_timeout = 5.0
    proto._encoder = MessageEncoder()
    proto._client_id = 1
    proto._decoder = MessageDecoder(is_request=False)
    proto._decoder.feed(frame_bytes)
    proto._lock = asyncio.Lock()
    proto._addr_suffix = lambda: ""
    return proto


async def test_read_response_small_message_stays_in_loop() -> None:
    """A small (sub-threshold) response must decode in-loop, not pay the thread hop."""
    frame_bytes = _build_empty_response_bytes()
    proto = _make_protocol_with_buffered_response(frame_bytes)

    loop_thread_id = threading.get_ident()
    decode_thread_ids: list[int] = []
    real_decode_bytes = proto._decoder.decode_bytes

    def _tracking_decode_bytes(data: Any) -> Any:
        decode_thread_ids.append(threading.get_ident())
        return real_decode_bytes(data)

    proto._decoder.decode_bytes = _tracking_decode_bytes

    await proto._read_response()
    assert decode_thread_ids, "decode_bytes was never called"
    for tid in decode_thread_ids:
        assert tid == loop_thread_id, (
            f"small response unexpectedly offloaded to thread {tid}; "
            f"expected in-loop decode on {loop_thread_id}"
        )


async def test_read_response_large_message_offloads_decode() -> None:
    """A large RowsResponse frame (above 256 KiB) must decode on a worker thread."""
    # ~40k rows x 4 cols ~= 1.3 MiB encoded, above the 256 KiB threshold.
    frame_bytes = _build_rows_response_bytes(n_rows=40_000, n_cols=4)
    assert len(frame_bytes) > 256 * 1024, (
        f"test fixture too small ({len(frame_bytes)} bytes); should be >256 KiB"
    )
    proto = _make_protocol_with_buffered_response(frame_bytes)

    loop_thread_id = threading.get_ident()
    decode_thread_ids: list[int] = []
    real_decode_bytes = proto._decoder.decode_bytes

    def _tracking_decode_bytes(data: Any) -> Any:
        decode_thread_ids.append(threading.get_ident())
        return real_decode_bytes(data)

    proto._decoder.decode_bytes = _tracking_decode_bytes

    response = await proto._read_response()
    assert isinstance(response, RowsResponse)
    assert len(response.rows) == 40_000

    assert decode_thread_ids, "decode_bytes was never called"
    for tid in decode_thread_ids:
        assert tid != loop_thread_id, (
            f"large RowsResponse decode ran on loop thread ({tid}); "
            f"must offload via asyncio.to_thread for multi-MiB payloads"
        )


async def test_read_response_threshold_is_documented_constant() -> None:
    """The decode threshold is a module-level constant tunable at a single site."""
    from dqliteclient import protocol as protocol_mod

    assert hasattr(protocol_mod, "_DECODE_OFFLOAD_THRESHOLD")
    threshold = protocol_mod._DECODE_OFFLOAD_THRESHOLD
    assert 64 * 1024 <= threshold <= 1024 * 1024


async def test_read_response_rejects_trailing_frame_after_terminal() -> None:
    """Hostile-server hardening: a trailing frame after a terminal response is rejected,
    and the check still fires under the offload decode path."""
    files_bytes = MessageEncoder().encode(FilesResponse(files={"main.db": b"\x00" * 64}))
    trailing = _build_empty_response_bytes()
    poisoned = files_bytes + trailing
    proto = _make_protocol_with_buffered_response(poisoned)

    from dqliteclient.exceptions import ProtocolError

    with pytest.raises(ProtocolError, match="extra response"):
        await proto._read_response()
