"""Protocol._read_response rejects extra frames after every terminal response (RowsResponse
excepted — its continuation frames span decode steps).

Without this, extra bytes from a buggy/malicious server get consumed as the next RPC's response,
surfacing a misleading error against an unrelated operation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import (
    EmptyResponse,
    LeaderResponse,
    ResultResponse,
    WelcomeResponse,
)
from dqlitewire.messages.responses import DbResponse, StmtResponse


@pytest.fixture
def protocol() -> DqliteProtocol:
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
    protocol: DqliteProtocol,
    first: object,
    second: object,
) -> None:
    """Two coalesced responses whose first is a non-Rows terminal must raise ProtocolError."""
    payload = first.encode() + second.encode()  # type: ignore[attr-defined]
    protocol._reader.read.return_value = payload  # type: ignore[attr-defined]

    with pytest.raises(ProtocolError, match="extra response"):
        await protocol._read_response()
