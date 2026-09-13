"""DqliteProtocol failure paths: FailureResponse hygiene, continuation caps, stray frames."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol, _failure_message
from dqlitewire.constants import ROW_PART_MARKER, ValueType
from dqlitewire.messages import (
    EmptyResponse,
    FailureResponse,
    LeaderResponse,
    ResultResponse,
    RowsResponse,
    WelcomeResponse,
)
from dqlitewire.messages.responses import DbResponse, StmtResponse
from dqlitewire.types import encode_uint64


class TestFailureMessage:
    @pytest.mark.parametrize(
        ("message", "addr_suffix", "expected"),
        [
            ("real msg", " to host:9001", "real msg to host:9001"),
            ("real msg", "", "real msg"),
            ("", " to host:9001", "(no diagnostic from server) to host:9001"),
            ("", "", "(no diagnostic from server)"),
            ("   ", " to host:9001", "(no diagnostic from server) to host:9001"),
            ("\t\n", "", "(no diagnostic from server)"),
        ],
    )
    def test_renders_message_with_address(
        self, message: str, addr_suffix: str, expected: str
    ) -> None:
        assert _failure_message(message, addr_suffix) == expected


@pytest.fixture
def hygiene_protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer)


@pytest.mark.asyncio
async def test_two_failures_in_a_row_raises_protocol_error(
    hygiene_protocol: DqliteProtocol,
) -> None:
    """Two failures in one read raise ProtocolError instead of buffering
    the second for cross-misattribution."""
    two_failures = (
        FailureResponse(code=19, message="first").encode()
        + FailureResponse(code=20, message="second").encode()
    )
    hygiene_protocol._reader.read = AsyncMock(side_effect=[two_failures, b""])
    with pytest.raises(ProtocolError, match="extra response"):
        await hygiene_protocol.exec_sql(db_id=1, sql="INSERT INTO t VALUES (1)")


@pytest.mark.asyncio
async def test_single_failure_raises_operational_error_not_protocol_error(
    hygiene_protocol: DqliteProtocol,
) -> None:
    """A single FailureResponse still surfaces as OperationalError; the
    check is gated on another frame being buffered."""
    from dqliteclient.exceptions import OperationalError

    one_failure = FailureResponse(code=19, message="constraint failed").encode()
    hygiene_protocol._reader.read = AsyncMock(side_effect=[one_failure, b""])
    with pytest.raises(OperationalError, match="constraint failed"):
        await hygiene_protocol.exec_sql(db_id=1, sql="INSERT INTO t VALUES (1)")


def _make_protocol() -> DqliteProtocol:
    return DqliteProtocol(MagicMock(), MagicMock(), timeout=5.0)


def test_failure_text_strips_u2028_line_separator() -> None:
    """U+2028 is legal in TEXT decoding but splits journald records."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="error executing query INJECTED LOG LINE",
    )
    rendered = proto._failure_text(response)
    assert " " not in rendered


def test_failure_text_strips_bidi_override() -> None:
    """U+202E RIGHT-TO-LEFT OVERRIDE could hide attacker segments."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="benign ‮malicious-rtl-content",
    )
    rendered = proto._failure_text(response)
    assert "‮" not in rendered


def test_failure_text_preserves_lf_for_multi_line_messages() -> None:
    """Display variant preserves LF for multi-line server diagnostics."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="line 1\nline 2",
    )
    rendered = proto._failure_text(response)
    assert "\n" in rendered


def test_failure_text_preserves_tab_for_columnar_diagnostics() -> None:
    """Tab is preserved by the display variant (the strict one escapes it)."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="col1\tcol2",
    )
    rendered = proto._failure_text(response)
    assert "\t" in rendered


def _make_proto() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    # Widen _read_timeout (300s) away from the 5s operator timeout so leakage is detectable.
    proto = DqliteProtocol(reader, writer, timeout=5.0)
    proto._read_timeout = 300.0
    return proto


@pytest.mark.asyncio
async def test_past_deadline_message_reports_overrun_not_read_timeout() -> None:
    proto = _make_proto()
    loop = asyncio.get_running_loop()
    past_deadline = loop.time() - 1.5  # ~1.5s past

    with pytest.raises(DqliteConnectionError) as ei:
        await proto._read_data(deadline=past_deadline)

    msg = str(ei.value)
    assert re.search(r"exceeded deadline by \d+\.\d+s", msg), msg
    # The widened _read_timeout (300.0) must not leak into the message.
    assert "300" not in msg, f"_read_timeout leaked into message; got {msg!r}"


@pytest.fixture
def terminal_protocol() -> DqliteProtocol:
    reader = AsyncMock(spec=asyncio.StreamReader)
    writer = MagicMock(spec=asyncio.StreamWriter)
    proto = DqliteProtocol(reader, writer, timeout=2.0)
    # Skip the handshake so _read_response works directly.
    proto._decoder._handshake_done = True
    proto._decoder._version = 1
    return proto


@pytest.mark.parametrize(
    "first,second",
    [
        (EmptyResponse(), EmptyResponse()),
        (
            ResultResponse(last_insert_id=1, rows_affected=1),
            ResultResponse(last_insert_id=2, rows_affected=2),
        ),
        (
            WelcomeResponse(heartbeat_timeout=15000),
            WelcomeResponse(heartbeat_timeout=15000),
        ),
        (
            LeaderResponse(node_id=1, address="h:9001"),
            LeaderResponse(node_id=2, address="h:9002"),
        ),
        (DbResponse(db_id=1), DbResponse(db_id=2)),
        (
            StmtResponse(db_id=1, stmt_id=1, num_params=0),
            StmtResponse(db_id=1, stmt_id=2, num_params=0),
        ),
    ],
)
@pytest.mark.asyncio
async def test_extra_frame_after_terminal_raises(
    terminal_protocol: DqliteProtocol,
    first: object,
    second: object,
) -> None:
    """Two coalesced responses whose first is a non-Rows terminal must raise ProtocolError."""
    payload = first.encode() + second.encode()  # type: ignore[attr-defined]
    terminal_protocol._reader.read.return_value = payload  # type: ignore[attr-defined]

    with pytest.raises(ProtocolError, match="extra response"):
        await terminal_protocol._read_response()


def _make_yield_protocol() -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=5.0)


@pytest.mark.asyncio
async def test_drain_continuations_yields_between_prefetched_frames() -> None:
    """Synchronously-decoded continuation frames must still yield to siblings each iteration."""
    proto = _make_yield_protocol()

    # 200 prefetched continuation frames + a final has_more=False frame.
    cont_frames = [
        RowsResponse(column_names=["x"], rows=[[i]], has_more=True) for i in range(1, 201)
    ]
    cont_frames.append(RowsResponse(column_names=["x"], rows=[[201]], has_more=False))
    cont_iter = iter(cont_frames)

    async def fake_read_continuation(deadline: float) -> RowsResponse:
        # No await: mirrors the StreamReader fast path with the next frame already buffered.
        return next(cont_iter)

    proto._read_continuation = fake_read_continuation

    sibling_ran = 0

    async def sibling() -> None:
        nonlocal sibling_ran
        while True:
            await asyncio.sleep(0)
            sibling_ran += 1

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)

    sibling_task = asyncio.create_task(sibling())
    try:
        # Schedule the sibling before the drain claims the loop.
        await asyncio.sleep(0)
        baseline = sibling_ran

        rows, _types = await proto._drain_continuations(initial, deadline=999999.0)

        during = sibling_ran - baseline
    finally:
        sibling_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sibling_task

    assert len(rows) == 202, "all frames must be drained"
    # >= 50 (vs 0 under the prior code) guards against a future yield-every-Nth refactor.
    assert during >= 50, (
        f"sibling ran only {during} times while drain processed 201 "
        "prefetched frames; cooperative yield missing"
    )


def _make_capped_protocol(max_continuation_frames: int = 2) -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(
        reader,
        writer,
        timeout=5.0,
        max_continuation_frames=max_continuation_frames,
    )


@pytest.mark.asyncio
async def test_drain_continuations_cap_is_inclusive() -> None:
    """Cap=2 allows initial + one continuation; a third frame must raise."""
    proto = _make_capped_protocol(max_continuation_frames=2)
    cont_frames = iter(
        [
            RowsResponse(column_names=["x"], rows=[[1]], has_more=True),
            RowsResponse(column_names=["x"], rows=[[2]], has_more=False),
        ]
    )

    async def fake_read_cont(deadline: float) -> RowsResponse:
        return next(cont_frames)

    proto._read_continuation = fake_read_cont

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)
    with pytest.raises(ProtocolError, match="max_continuation_frames"):
        await proto._drain_continuations(initial, deadline=0.0)


@pytest.mark.asyncio
async def test_drain_continuations_cap_at_exact_limit_accepted() -> None:
    """Cap=2 with exactly 2 decoded frames must succeed."""
    proto = _make_capped_protocol(max_continuation_frames=2)
    cont_frames = iter([RowsResponse(column_names=["x"], rows=[[1]], has_more=False)])

    async def fake_read_cont(deadline: float) -> RowsResponse:
        return next(cont_frames)

    proto._read_continuation = fake_read_cont

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)
    rows, _types = await proto._drain_continuations(initial, deadline=0.0)
    assert rows == [[0], [1]]


def _make_proto_with_address(address: str) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = address
    return proto


@pytest.mark.asyncio
async def test_continuation_failure_arm_preserves_addr_suffix_at_wire_cap() -> None:
    """A near-wire-cap FAILURE mid-continuation: OperationalError.message must
    still end with the peer-address suffix."""
    from dqlitewire.exceptions import ServerFailure as _WireServerFailure

    proto = _make_proto_with_address("host-c:19003")

    # 65000-char message (> 1024 display cap) so without pre-truncation the
    # suffix is pushed off the display.
    class _StubDecoder:
        def decode_continuation(
            self,
        ) -> RowsResponse | _WireServerFailure | None:
            raise _WireServerFailure(code=1001, message="X" * 65000)

        def feed(self, data: bytes) -> None:
            pass

        def pending_frame_size(self) -> int:
            return 0

    proto._decoder = _StubDecoder()  # type: ignore[assignment]

    async def _stub_read_data(deadline: float) -> bytes:
        return b""

    proto._read_data = _stub_read_data

    proto._timeout = 1.0
    proto._read_timeout = 1.0
    proto._max_continuation_frames = 100

    with pytest.raises(OperationalError) as exc_info:
        await asyncio.wait_for(
            proto._read_continuation(deadline=999_999.0),
            timeout=1.0,
        )

    err = exc_info.value
    rendered = str(err)
    assert rendered.endswith(" to host-c:19003"), (
        f"Addr suffix dropped after display truncation: ...{rendered[-200:]!r}"
    )
    assert "[truncated," in rendered, (
        f"Expected truncation marker in display message: {rendered[:200]!r}..."
    )


@pytest.mark.asyncio
async def test_continuation_failure_arm_short_message_keeps_suffix() -> None:
    """Short FAILURE message: suffix is appended verbatim, no truncation."""
    from dqlitewire.exceptions import ServerFailure as _WireServerFailure

    proto = _make_proto_with_address("host-d:9000")

    class _StubDecoder:
        def decode_continuation(self) -> _WireServerFailure | None:
            raise _WireServerFailure(code=42, message="boom")

        def feed(self, data: bytes) -> None:
            pass

        def pending_frame_size(self) -> int:
            return 0

    proto._decoder = _StubDecoder()  # type: ignore[assignment]

    async def _stub_read_data(deadline: float) -> bytes:
        return b""

    proto._read_data = _stub_read_data

    proto._timeout = 1.0
    proto._read_timeout = 1.0
    proto._max_continuation_frames = 100

    with pytest.raises(OperationalError) as exc_info:
        await asyncio.wait_for(
            proto._read_continuation(deadline=999_999.0),
            timeout=1.0,
        )

    rendered = str(exc_info.value)
    assert rendered.endswith(" to host-d:9000")
    assert "[truncated," not in rendered


@pytest.fixture
def empty_message_protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    p = DqliteProtocol(reader, writer)
    return p


@pytest.mark.parametrize("code", [1, 19, 10250])
@pytest.mark.parametrize("message", ["", "   ", "\t\n"])
async def test_exec_sql_empty_message_renders_placeholder(
    empty_message_protocol: DqliteProtocol, code: int, message: str
) -> None:
    empty_message_protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
        code=code, message=message
    ).encode()
    with pytest.raises(OperationalError) as exc_info:
        await empty_message_protocol.exec_sql(1, "SELECT 1")
    assert "(no diagnostic from server)" in str(exc_info.value)
    assert exc_info.value.code == code


@pytest.mark.parametrize("code", [1, 19, 10250])
async def test_query_sql_empty_message_renders_placeholder(
    empty_message_protocol: DqliteProtocol, code: int
) -> None:
    empty_message_protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
        code=code, message=""
    ).encode()
    with pytest.raises(OperationalError) as exc_info:
        await empty_message_protocol.query_sql(1, "SELECT 1")
    assert "(no diagnostic from server)" in str(exc_info.value)
    assert exc_info.value.code == code


@pytest.fixture
def mid_stream_protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=1.0, address="test:9001")


class TestServerFailureMidStreamClassification:
    async def test_mid_stream_failure_raises_operational_error_with_code(
        self, mid_stream_protocol: DqliteProtocol
    ) -> None:
        # Initial frame: row with PART marker signaling more to come.
        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        part_marker = encode_uint64(ROW_PART_MARKER)
        frame_with_part = initial[:-8] + part_marker

        # SQLITE_IOERR_NOT_LEADER = 10250.
        failure = FailureResponse(code=10250, message="not leader")
        failure_frame = failure.encode()

        mid_stream_protocol._reader.read = AsyncMock(
            side_effect=[frame_with_part + failure_frame, b""]
        )

        with pytest.raises(OperationalError) as exc_info:
            await mid_stream_protocol.query_sql(1, "SELECT 1")

        assert exc_info.value.code == 10250
        assert "not leader" in exc_info.value.message

    async def test_mid_stream_non_leader_failure_keeps_connection_usable(
        self, mid_stream_protocol: DqliteProtocol
    ) -> None:
        """After a mid-stream non-leader FailureResponse the buffer is not
        poisoned, so the next request on the same connection decodes cleanly."""
        from dqlitewire.messages import LeaderResponse

        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        part_marker = encode_uint64(ROW_PART_MARKER)
        frame_with_part = initial[:-8] + part_marker

        # SQLITE_CONSTRAINT = 19.
        failure_frame = FailureResponse(code=19, message="CHECK constraint failed").encode()

        followup_frame = LeaderResponse(node_id=1, address="127.0.0.1:9001").encode()

        mid_stream_protocol._reader.read = AsyncMock(
            side_effect=[frame_with_part + failure_frame, followup_frame, b""]
        )

        with pytest.raises(OperationalError) as exc_info:
            await mid_stream_protocol.query_sql(1, "SELECT 1")
        assert exc_info.value.code == 19

        assert not mid_stream_protocol._decoder.is_poisoned
        leader = await mid_stream_protocol._read_response()
        assert isinstance(leader, LeaderResponse)
        assert leader.address == "127.0.0.1:9001"

    async def test_mid_stream_failure_not_raised_as_protocol_error(
        self, mid_stream_protocol: DqliteProtocol
    ) -> None:
        """Regression guard: the raised exception must not be ProtocolError
        (except via OperationalError's own MRO)."""
        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        part_marker = encode_uint64(ROW_PART_MARKER)
        frame_with_part = initial[:-8] + part_marker

        failure_frame = FailureResponse(code=5, message="db locked").encode()
        mid_stream_protocol._reader.read = AsyncMock(
            side_effect=[frame_with_part + failure_frame, b""]
        )

        with pytest.raises(OperationalError) as exc_info:
            await mid_stream_protocol.query_sql(1, "SELECT 1")

        assert not isinstance(exc_info.value, ProtocolError)
