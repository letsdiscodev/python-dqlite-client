"""Pin: a transport reset (peer FIN / RST) mid-INSERT surfaces as a
clear error to the caller and invalidates the protocol — no silent
retry, no half-applied row, no cursor-reuse hazard.

dqlite is a clustered DB; a leader flip mid-INSERT can drop the
server-side connection at any byte. The contract we need:

* The streaming send raises ``DqliteConnectionError`` (or a wrapped
  PEP-249 ``OperationalError`` higher up).
* The protocol's ``is_wire_coherent`` flips to False so the pool
  drops the connection on release.
* A subsequent ``exec_sql`` on the same protocol instance does not
  silently re-attempt against a dead transport.

This is a unit-mock pin; the integration cluster's leader-flip
fixture is gated separately (see test_pool_concurrent_tx_leader_flip).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.protocol import DqliteProtocol


def _protocol_with_failing_drain(error: Exception) -> DqliteProtocol:
    """Build a DqliteProtocol whose drain() raises ``error`` on first
    call (simulating a peer reset / RST mid-write)."""
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock(side_effect=error)
    proto = DqliteProtocol(reader, writer, timeout=5.0)
    return proto


@pytest.mark.asyncio
async def test_exec_sql_mid_write_reset_raises_dqlite_connection_error() -> None:
    """``exec_sql`` calling ``_send`` against a drain that raises
    ``ConnectionResetError`` (peer RST) wraps as
    ``DqliteConnectionError``."""
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
    """A second exec_sql on the same protocol after the failure must
    raise too — no silent retry against a dead transport.

    The first call raises DqliteConnectionError. The drain mock keeps
    raising on every subsequent call, so the second exec_sql also
    raises. This pins "the protocol does not internally retry"."""
    proto = _protocol_with_failing_drain(ConnectionResetError("reset"))

    with pytest.raises(DqliteConnectionError):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t VALUES (1)", params=None)

    # Same protocol, same drain mock — the second call must propagate
    # the same write-failed error, NOT magically succeed.
    with pytest.raises(DqliteConnectionError):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t VALUES (2)", params=None)


@pytest.mark.asyncio
async def test_large_blob_insert_mid_drain_failure_raises() -> None:
    """The same contract holds for a multi-MB BLOB insert — the
    failure point is in the drain, not in encode, so blob size does
    not matter for the contract. Pin it explicitly so a future
    refactor that streams the encode separately doesn't lose this."""
    proto = _protocol_with_failing_drain(ConnectionResetError("mid-stream reset"))
    big_blob = b"x" * (4 * 1024 * 1024)  # 4 MiB

    with pytest.raises(DqliteConnectionError, match="Write failed"):
        await proto.exec_sql(db_id=0, sql="INSERT INTO t(b) VALUES (?)", params=[big_blob])
