"""DqliteProtocol.dump runs the multi-MiB FilesResponse.decode_body on a worker thread via
asyncio.to_thread so the per-file materialise chain does not freeze the event loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire import MessageDecoder, MessageEncoder
from dqlitewire.messages import requests as wire_requests
from dqlitewire.messages.responses import FilesResponse

pytestmark = pytest.mark.asyncio


def _build_dump_response_bytes(files: dict[str, bytes]) -> bytes:
    encoder = MessageEncoder()
    response = FilesResponse(files=files)
    return encoder.encode(response)


def _make_protocol_with_buffered_response(frame_bytes: bytes) -> DqliteProtocol:
    """Build a DqliteProtocol with the response pre-buffered so _read_data is never called."""
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._writer = MagicMock()
    proto._writer.write = MagicMock()
    proto._writer.drain = AsyncMock()
    proto._timeout = 5.0
    proto._read_timeout = 5.0
    proto._encoder = MessageEncoder()
    proto._client_id = 1
    # MessageDecoder(is_request=False) starts handshake-done so we can decode immediately.
    proto._decoder = MessageDecoder(is_request=False)
    proto._decoder.feed(frame_bytes)
    proto._lock = asyncio.Lock()
    proto._addr_suffix = lambda: ""
    return proto


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
