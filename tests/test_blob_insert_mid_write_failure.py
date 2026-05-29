"""A transport reset mid-INSERT raises DqliteConnectionError and does not silently retry.

A leader flip mid-INSERT can drop the server-side connection at any byte.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.protocol import DqliteProtocol


def _protocol_with_failing_drain(error: Exception) -> DqliteProtocol:
    """Build a DqliteProtocol whose drain() raises error (peer reset mid-write)."""
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock(side_effect=error)
    proto = DqliteProtocol(reader, writer, timeout=5.0)
    return proto


@pytest.mark.asyncio
async def test_exec_sql_mid_write_reset_raises_dqlite_connection_error() -> None:
    """A drain ConnectionResetError (peer RST) wraps as DqliteConnectionError."""
    proto = _protocol_with_failing_drain(ConnectionResetError("connection reset"))

    with pytest.raises(DqliteConnectionError, match="Write failed"):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t(b) VALUES (?)", params=[b"x" * 100])


@pytest.mark.asyncio
async def test_exec_sql_mid_write_broken_pipe_raises_dqlite_connection_error() -> None:
    """BrokenPipeError (FIN-then-write) also surfaces uniformly."""
    proto = _protocol_with_failing_drain(BrokenPipeError("broken pipe"))

    with pytest.raises(DqliteConnectionError, match="Write failed"):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t(b) VALUES (?)", params=[b"x" * 100])


@pytest.mark.asyncio
async def test_exec_sql_mid_write_oserror_raises_dqlite_connection_error() -> None:
    """Generic OSError (host unreachable mid-stream) surfaces uniformly."""
    proto = _protocol_with_failing_drain(OSError("transport torn down"))

    with pytest.raises(DqliteConnectionError, match="Write failed"):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t(b) VALUES (?)", params=[b"x" * 100])


@pytest.mark.asyncio
async def test_exec_sql_mid_write_does_not_silently_retry() -> None:
    """A second exec_sql after the failure also raises: no silent retry on a dead transport."""
    proto = _protocol_with_failing_drain(ConnectionResetError("reset"))

    with pytest.raises(DqliteConnectionError):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t VALUES (1)", params=None)

    with pytest.raises(DqliteConnectionError):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t VALUES (2)", params=None)


@pytest.mark.asyncio
async def test_large_blob_insert_mid_drain_failure_raises() -> None:
    """The contract holds for a multi-MB BLOB: failure is in drain, not encode."""
    proto = _protocol_with_failing_drain(ConnectionResetError("mid-stream reset"))
    big_blob = b"x" * (4 * 1024 * 1024)  # 4 MiB

    with pytest.raises(DqliteConnectionError, match="Write failed"):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t(b) VALUES (?)", params=[big_blob])
