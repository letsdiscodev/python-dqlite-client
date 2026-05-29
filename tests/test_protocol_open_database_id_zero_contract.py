"""Pin: ``DqliteProtocol.open_database`` enforces the upstream
"first DB on a fresh connection gets db_id=0" contract, catching a
bad id at the OPEN site before it propagates to subsequent RPCs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire import WIRE_DECODE_FAILED_PREFIX


@pytest.mark.asyncio
async def test_open_database_rejects_nonzero_db_id() -> None:
    from dqlitewire.messages import DbResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    bad = DbResponse(db_id=42)
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=bad)),
        pytest.raises(ProtocolError) as exc_info,
    ):
        await protocol.open_database("default")

    msg = str(exc_info.value)
    assert "expected 0" in msg
    assert WIRE_DECODE_FAILED_PREFIX in msg
    assert "db_id=42" in msg


@pytest.mark.asyncio
async def test_open_database_happy_path_returns_zero() -> None:
    from dqlitewire.messages import DbResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    ok = DbResponse(db_id=0)
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=ok)),
    ):
        db_id = await protocol.open_database("default")
    assert db_id == 0
