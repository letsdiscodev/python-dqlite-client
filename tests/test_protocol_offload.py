"""DqliteProtocol offloads large encodes and decodes to a worker thread."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import protocol as protocol_mod
from dqliteclient.protocol import DqliteProtocol
from dqlitewire import MessageDecoder, MessageEncoder
from dqlitewire.constants import ValueType
from dqlitewire.messages import requests as wire_requests
from dqlitewire.messages.responses import EmptyResponse, FilesResponse, RowsResponse


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


def test_estimate_counts_prepare_request_sql() -> None:
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    big_sql = "x" * (threshold)  # *4 in the estimate → comfortably over
    request = wire_requests.PrepareRequest(db_id=1, sql=big_sql)
    estimate = protocol_mod._estimate_request_body_size(request)
    assert estimate >= threshold, (
        f"PrepareRequest with {len(big_sql)}-char SQL estimated to "
        f"{estimate}, below the {threshold}-byte gate; the estimator "
        f"must count the ``sql`` field, not just ``params``."
    )


def test_estimate_counts_exec_sql_inline_literal() -> None:
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    big_sql = "INSERT INTO t VALUES " + "(1)," * (threshold // 4)
    request = wire_requests.ExecSqlRequest(db_id=1, sql=big_sql, params=[])
    estimate = protocol_mod._estimate_request_body_size(request)
    assert estimate >= threshold, (
        f"ExecSqlRequest with a {len(big_sql)}-char inline-literal SQL "
        f"estimated to {estimate}, below the {threshold}-byte gate."
    )


def test_estimate_small_sql_stays_below_threshold() -> None:
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    request = wire_requests.PrepareRequest(db_id=1, sql="SELECT 1")
    estimate = protocol_mod._estimate_request_body_size(request)
    assert estimate < threshold, (
        f"small PrepareRequest estimated to {estimate}; must stay below "
        f"the {threshold}-byte gate so heartbeat-class statements do not "
        f"pay the thread-hop cost."
    )


@pytest.mark.asyncio
async def test_prepare_request_large_sql_dispatches_via_to_thread() -> None:
    proto = _make_protocol_with_mock_writer()
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    big_sql = "x" * threshold
    request = wire_requests.PrepareRequest(db_id=1, sql=big_sql)

    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(request)

    assert len(to_thread_calls) == 1, (
        f"large-SQL PrepareRequest must offload the encode to "
        f"asyncio.to_thread; got {len(to_thread_calls)} hops."
    )
    assert proto._writer.write.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_small_prepare_request_stays_in_loop() -> None:
    """A small ``PrepareRequest`` must NOT offload."""
    proto = _make_protocol_with_mock_writer()

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    request = wire_requests.PrepareRequest(db_id=1, sql="SELECT 1")
    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(request)

    assert to_thread_calls == [], (
        f"small PrepareRequest unexpectedly offloaded: {to_thread_calls!r}"
    )


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


def _build_dump_response_bytes(files: dict[str, bytes]) -> bytes:
    encoder = MessageEncoder()
    response = FilesResponse(files=files)
    return encoder.encode(response)


async def test_dump_decode_runs_on_worker_thread() -> None:
    """FilesResponse.decode_body must run off the loop thread."""
    files = {"main.db": b"\x00" * (256 * 1024)}  # 256 KiB payload
    frame_bytes = _build_dump_response_bytes(files)
    proto = _make_protocol_with_buffered_response(frame_bytes)

    loop_thread_id = threading.get_ident()
    decode_thread_ids: list[int] = []

    real_decode_bytes = proto._decoder.decode_bytes

    def _tracking_decode_bytes(data: Any) -> Any:
        decode_thread_ids.append(threading.get_ident())
        return real_decode_bytes(data)

    proto._decoder.decode_bytes = _tracking_decode_bytes

    result = await proto.dump("main")
    assert result == files

    assert decode_thread_ids, "decode_bytes was never called"
    for tid in decode_thread_ids:
        assert tid != loop_thread_id, (
            f"FilesResponse.decode_body ran on the loop thread "
            f"({tid}); it must offload via asyncio.to_thread to "
            f"avoid freezing the loop on multi-MiB dump payloads."
        )


async def test_dump_rejects_trailing_frame_pre_offload() -> None:
    """The terminal-frame hardening must fire on-loop before the off-loop decode begins."""
    files = {"main.db": b"\x00" * 64}
    frame_bytes = _build_dump_response_bytes(files)
    # Two coalesced frames simulate a hostile server appending extra bytes.
    trailing_frame = _build_dump_response_bytes({"trailing.db": b"\x00" * 8})
    poisoned_bytes = frame_bytes + trailing_frame

    proto = _make_protocol_with_buffered_response(poisoned_bytes)

    decode_calls: list[Any] = []
    real_decode_bytes = proto._decoder.decode_bytes

    def _track_decode(data: Any) -> Any:
        decode_calls.append(data)
        return real_decode_bytes(data)

    proto._decoder.decode_bytes = _track_decode

    from dqliteclient.exceptions import ProtocolError

    with pytest.raises(ProtocolError, match="extra response"):
        await proto.dump("main")
    assert decode_calls == [], (
        "hostile-server trailing-frame check must fire on-loop "
        "BEFORE off-loop decode; saw decode_bytes invoked"
    )


async def test_dump_handles_failure_response_off_loop() -> None:
    """A FailureResponse to DumpRequest decodes off-loop and surfaces as OperationalError."""
    from dqlitewire.messages.responses import FailureResponse

    encoder = MessageEncoder()
    failure = FailureResponse(code=1, message="synthetic failure")
    frame_bytes = encoder.encode(failure)
    proto = _make_protocol_with_buffered_response(frame_bytes)

    from dqliteclient.exceptions import OperationalError

    with pytest.raises(OperationalError, match="synthetic failure"):
        await proto.dump("main")


async def test_dump_runs_request_send_through_send_request_helper() -> None:
    """dump routes its send through the threshold-gated _send_request helper, not direct _send."""
    files = {"main.db": b"\x00" * 64}
    frame_bytes = _build_dump_response_bytes(files)
    proto = _make_protocol_with_buffered_response(frame_bytes)

    send_request_calls: list[Any] = []
    real_send_request = proto._send_request

    async def _track_send_request(request: Any) -> None:
        send_request_calls.append(request)
        await real_send_request(request)

    proto._send_request = _track_send_request
    await proto.dump("main")

    assert len(send_request_calls) == 1
    assert isinstance(send_request_calls[0], wire_requests.DumpRequest)
