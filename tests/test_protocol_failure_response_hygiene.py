"""Pin: a hostile/buggy server emitting two FailureResponses for one
request triggers ``ProtocolError`` and invalidation rather than
silently misattributing the second failure to the next user request.

dqlite's wire protocol is strictly one-request-one-response. A server
that sends two FailureResponses (or any extra frame) for a single
request leaves the second buffered in the decoder; the next user
request would consume it as if it were that request's response —
cross-request misattribution.

The fix: in ``_read_response``, if the just-decoded message is a
``FailureResponse`` AND the decoder still has another frame
buffered, raise ``ProtocolError``. ``_run_protocol``'s
``ProtocolError`` arm calls ``_invalidate`` and the pool discards
the slot.
"""

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
    """Inject two failures in one TCP read; first request raises
    ProtocolError (which invalidates) instead of consuming and leaving
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
    """Negative pin: a single FailureResponse (the conforming case)
    still surfaces as the existing OperationalError, NOT as
    ProtocolError. The hostile-server check is gated on the buffer
    having ANOTHER frame queued."""
    from dqliteclient.exceptions import OperationalError

    one_failure = FailureResponse(code=19, message="constraint failed").encode()
    protocol._reader.read = AsyncMock(side_effect=[one_failure, b""])
    with pytest.raises(OperationalError, match="constraint failed"):
        await protocol.exec_sql(db_id=1, sql="INSERT INTO t VALUES (1)")
