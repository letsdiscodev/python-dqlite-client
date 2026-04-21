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
    proto._handshake_done = True
    return proto


class TestInterrupt:
    async def test_interrupt_drains_to_empty_response(self, protocol: DqliteProtocol) -> None:
        """Interrupt sends the request and consumes messages until
        EmptyResponse arrives.
        """
        protocol._reader.read = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[EmptyResponse().encode(), b""]
        )
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
            rows=[(1,)],
            row_types=[(ValueType.INTEGER,)],
            has_more=False,
        ).encode()
        empty = EmptyResponse().encode()
        protocol._reader.read = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[rows + empty, b""]
        )
        await protocol.interrupt(db_id=1)

    async def test_interrupt_raises_operational_error_on_failure(
        self, protocol: DqliteProtocol
    ) -> None:
        failure = FailureResponse(code=5, message="interrupted").encode()
        protocol._reader.read = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[failure, b""]
        )
        with pytest.raises(OperationalError) as exc_info:
            await protocol.interrupt(db_id=1)
        assert exc_info.value.code == 5

    async def test_interrupt_raises_protocol_error_on_unexpected_message(
        self, protocol: DqliteProtocol
    ) -> None:
        """An unexpected message type mid-drain (e.g. a DbResponse) is a
        stream desync; raise ProtocolError.
        """
        from dqlitewire.messages import DbResponse

        unexpected = DbResponse(db_id=42).encode()
        protocol._reader.read = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[unexpected, b""]
        )
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.interrupt(db_id=1)
