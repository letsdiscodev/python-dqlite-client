"""Unit tests for ``DqliteProtocol.interrupt``."""

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
        """Interrupt sends the request and consumes messages until EmptyResponse."""
        protocol._reader.read = AsyncMock(side_effect=[EmptyResponse().encode(), b""])
        await protocol._interrupt(db_id=1)
        protocol._writer.write.assert_called()  # type: ignore[attr-defined]

    async def test_interrupt_drains_result_then_empty_consumes_both(
        self, protocol: DqliteProtocol
    ) -> None:
        """Wire carries ``[RESULT, EMPTY]`` (gateway.c:952-960 re-dispatch
        emits a separate EMPTY ack); the drain loop must consume both."""
        from dqlitewire.messages import ResultResponse

        result = ResultResponse(last_insert_id=42, rows_affected=1).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(side_effect=[result + empty, b""])
        await protocol._interrupt(db_id=1)
        assert not protocol._decoder.has_message(), (
            "EMPTY following RESULT must be consumed by the drain loop"
        )

    async def test_interrupt_drains_result_then_empty_separate_reads(
        self, protocol: DqliteProtocol
    ) -> None:
        """Same gateway.c race, but RESULT and EMPTY arrive in separate
        ``reader.read()`` calls (no TCP coalescing); drain consumes both."""
        from dqlitewire.messages import ResultResponse

        result = ResultResponse(last_insert_id=42, rows_affected=1).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(side_effect=[result, empty, b""])
        await protocol._interrupt(db_id=1)
        assert not protocol._decoder.has_message()

    async def test_interrupt_drain_terminal_rows_then_empty(self, protocol: DqliteProtocol) -> None:
        """Terminal ROWS (has_more=False) followed by EMPTY: drain consumes
        both (gateway.c re-dispatch when ``g->req == NULL`` at INTERRUPT)."""
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
        """An in-flight RowsResponse after INTERRUPT is dropped; drain continues to EMPTY."""
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
        """Drain honours the same max_continuation_frames cap as the query
        path, so a slow-dripping server cannot pin a client past one deadline."""
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

        # Four in-flight frames arrive before EmptyResponse; cap = 3 trips on frame 4.
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
        """max_continuation_frames=None: bound only by the operation deadline."""
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
        await proto._interrupt(db_id=1)

    async def test_interrupt_raises_protocol_error_on_unexpected_message(
        self, protocol: DqliteProtocol
    ) -> None:
        """An unexpected message type mid-drain is a stream desync; raise ProtocolError."""
        from dqlitewire.messages import DbResponse

        unexpected = DbResponse(db_id=42).encode()
        protocol._reader.read = AsyncMock(side_effect=[unexpected, b""])
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol._interrupt(db_id=1)
