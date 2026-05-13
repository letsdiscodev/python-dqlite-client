"""Unit tests for ``DqliteProtocol.interrupt``.

The INTERRUPT wire message was defined and golden-tested but never
sent by the client layer — parity gap with go-dqlite and the C
client. Add the primitive so a future streaming-cursor API (and
cancel paths) can invoke it.

Our current ``query_raw_typed`` fully drains continuation frames
before returning, so there is no built-in call site yet. The
primitive is exposed on ``DqliteProtocol`` as a building block.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.constants import ValueType
from dqlitewire.messages import EmptyResponse, FailureResponse, RowsResponse


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    proto = DqliteProtocol(reader, writer, timeout=1.0, address="test:9001")
    return proto


class TestInterrupt:
    async def test_interrupt_drains_to_empty_response(self, protocol: DqliteProtocol) -> None:
        """Interrupt sends the request and consumes messages until
        EmptyResponse arrives.
        """
        protocol._reader.read = AsyncMock(side_effect=[EmptyResponse().encode(), b""])
        await protocol._interrupt(db_id=1)
        # Write was issued.
        protocol._writer.write.assert_called()  # type: ignore[attr-defined]

    async def test_interrupt_drains_result_then_empty_consumes_both(
        self, protocol: DqliteProtocol
    ) -> None:
        """An INTERRUPT landing AFTER an EXEC's done-callback has nulled
        the server-side ``g->req`` re-dispatches via ``handle_interrupt``
        (gateway.c:952-960), which emits a separate ``EmptyResponse`` for
        the interrupt itself. The wire then carries ``[RESULT, EMPTY]``
        — RESULT from the just-completed EXEC, EMPTY for the
        INTERRUPT ack. The drain loop must consume BOTH and leave the
        decoder buffer empty for the next RPC. Mirrors Go's
        ``Protocol.Interrupt`` loop-till-ResponseEmpty discipline.
        """
        from dqlitewire.messages import ResultResponse

        result = ResultResponse(last_insert_id=42, rows_affected=1).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(side_effect=[result + empty, b""])
        await protocol._interrupt(db_id=1)
        # Decoder buffer must be fully drained: no trailing EMPTY left
        # to poison the next RPC's read.
        assert not protocol._decoder.has_message(), (
            "EMPTY following RESULT must be consumed by the drain loop"
        )

    async def test_interrupt_drains_result_then_empty_separate_reads(
        self, protocol: DqliteProtocol
    ) -> None:
        """Same gateway.c race as the prior test, but RESULT and EMPTY
        arrive in separate ``reader.read()`` calls (no TCP coalescing).
        The drain loop must still consume both. This is the most
        plausible wire shape — the server emits the prior RPC's
        terminal first, then the EMPTY for the INTERRUPT ack a small
        amount of wall-clock time later.
        """
        from dqlitewire.messages import ResultResponse

        result = ResultResponse(last_insert_id=42, rows_affected=1).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(side_effect=[result, empty, b""])
        await protocol._interrupt(db_id=1)
        assert not protocol._decoder.has_message()

    async def test_interrupt_drain_terminal_rows_then_empty(self, protocol: DqliteProtocol) -> None:
        """A terminal ROWS frame (has_more=False) immediately followed
        by EMPTY: drain loop must consume both. Wire shape covers the
        case where the in-flight QUERY emitted its last continuation
        frame just before the INTERRUPT dispatched a separate EMPTY
        (gateway.c re-dispatch path via ``handle_interrupt`` when
        ``g->req == NULL`` at INTERRUPT-read time).
        """
        rows_terminal = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(side_effect=[rows_terminal + empty, b""])
        await protocol._interrupt(db_id=1)
        assert not protocol._decoder.has_message()

    async def test_interrupt_swallows_in_flight_rows(self, protocol: DqliteProtocol) -> None:
        """A RowsResponse landing after INTERRUPT (in-flight from before
        the server processed the interrupt) is dropped; the drain loop
        continues until EmptyResponse.
        """
        rows = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(side_effect=[rows + empty, b""])
        await protocol._interrupt(db_id=1)

    async def test_interrupt_raises_operational_error_on_failure(
        self, protocol: DqliteProtocol
    ) -> None:
        failure = FailureResponse(code=5, message="interrupted").encode()
        protocol._reader.read = AsyncMock(side_effect=[failure, b""])
        with pytest.raises(OperationalError) as exc_info:
            await protocol._interrupt(db_id=1)
        assert exc_info.value.code == 5

    async def test_interrupt_drain_respects_max_continuation_frames(self) -> None:
        """The drain loop must honour the same max_continuation_frames
        cap that _drain_continuations uses on the query path.
        Otherwise a slow-dripping server answering an INTERRUPT with
        many small RowsResponse frames can pin a client within a
        single operation deadline.
        """
        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        proto = DqliteProtocol(
            reader,
            writer,
            timeout=10.0,
            address="test:9001",
            max_continuation_frames=3,
        )

        # Four in-flight frames arrive before EmptyResponse; cap = 3
        # trips on frame 4. RowsResponse with has_more=False is the
        # canonical "done marker" shape the drain loop already swallows.
        rows_frame = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        empty = EmptyResponse().encode()
        proto._reader.read = AsyncMock(side_effect=[rows_frame * 4 + empty, b""])
        with pytest.raises(ProtocolError, match="max_continuation_frames"):
            await proto._interrupt(db_id=1)

    async def test_interrupt_drain_no_cap_when_governor_unset(self) -> None:
        """max_continuation_frames=None restores the existing behaviour
        (bound only by the operation deadline). Regression guard for
        callers that opt out of the cap."""
        reader = AsyncMock()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        proto = DqliteProtocol(
            reader,
            writer,
            timeout=10.0,
            address="test:9001",
            max_continuation_frames=None,
        )

        rows_frame = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        empty = EmptyResponse().encode()
        proto._reader.read = AsyncMock(side_effect=[rows_frame * 10 + empty, b""])
        # No raise: deadline-only behaviour preserved.
        await proto._interrupt(db_id=1)

    async def test_interrupt_raises_protocol_error_on_unexpected_message(
        self, protocol: DqliteProtocol
    ) -> None:
        """An unexpected message type mid-drain (e.g. a DbResponse) is a
        stream desync; raise ProtocolError.
        """
        from dqlitewire.messages import DbResponse

        unexpected = DbResponse(db_id=42).encode()
        protocol._reader.read = AsyncMock(side_effect=[unexpected, b""])
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol._interrupt(db_id=1)
