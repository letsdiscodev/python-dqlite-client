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
        await protocol.interrupt(db_id=1)
        # Write was issued.
        protocol._writer.write.assert_called()  # type: ignore[attr-defined]

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
        await protocol.interrupt(db_id=1)

    async def test_interrupt_raises_operational_error_on_failure(
        self, protocol: DqliteProtocol
    ) -> None:
        failure = FailureResponse(code=5, message="interrupted").encode()
        protocol._reader.read = AsyncMock(side_effect=[failure, b""])
        with pytest.raises(OperationalError) as exc_info:
            await protocol.interrupt(db_id=1)
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
            await proto.interrupt(db_id=1)

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
        await proto.interrupt(db_id=1)

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
            await protocol.interrupt(db_id=1)
