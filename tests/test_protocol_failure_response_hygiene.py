"""A server emitting two FailureResponses for one request must raise
``ProtocolError`` (which invalidates the slot) rather than leave the second
buffered for the next request to misconsume — the wire is one-request-one-response."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer)


@pytest.mark.asyncio
async def test_two_failures_in_a_row_raises_protocol_error(
    protocol: DqliteProtocol,
) -> None:
    """Two failures in one read raise ProtocolError instead of buffering
    the second for cross-misattribution."""
    two_failures = (
        FailureResponse(code=19, message="first").encode()
        + FailureResponse(code=20, message="second").encode()
    )
    protocol._reader.read = AsyncMock(side_effect=[two_failures, b""])
    with pytest.raises(ProtocolError, match="extra response"):
        await protocol.exec_sql(db_id=1, sql="INSERT INTO t VALUES (1)")


@pytest.mark.asyncio
async def test_single_failure_raises_operational_error_not_protocol_error(
    protocol: DqliteProtocol,
) -> None:
    """A single FailureResponse still surfaces as OperationalError; the
    check is gated on another frame being buffered."""
    from dqliteclient.exceptions import OperationalError

    one_failure = FailureResponse(code=19, message="constraint failed").encode()
    protocol._reader.read = AsyncMock(side_effect=[one_failure, b""])
    with pytest.raises(OperationalError, match="constraint failed"):
        await protocol.exec_sql(db_id=1, sql="INSERT INTO t VALUES (1)")
