"""Per-flag isolation for the pool-reset predicate: the lone
_savepoint_implicit_begin=True case (internally inconsistent, but the
predicate must still defend against it).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.connection import _TRANSACTION_ROLLBACK_SQL
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_reset_fires_when_only_savepoint_implicit_begin_true() -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = DqliteConnection("localhost:9001")
    with (
        patch.object(DqliteConnection, "is_connected", new=True),
        patch("dqliteclient.pool._socket_looks_dead", return_value=False),
    ):
        conn._in_transaction = False
        conn._savepoint_stack = []
        conn._savepoint_implicit_begin = True
        conn._has_untracked_savepoint = False

        executed: list[str] = []

        async def fake_execute(sql: str) -> object:
            executed.append(sql)
            return None

        with patch.object(conn, "execute", new=fake_execute):
            result = await pool._reset_connection(conn)

    assert executed == [_TRANSACTION_ROLLBACK_SQL]
    assert result is True


@pytest.mark.asyncio
async def test_reset_skips_when_all_four_flags_false() -> None:
    """Negative pin: with every flag clean, no ROLLBACK is issued."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = DqliteConnection("localhost:9001")
    with (
        patch.object(DqliteConnection, "is_connected", new=True),
        patch("dqliteclient.pool._socket_looks_dead", return_value=False),
    ):
        conn._in_transaction = False
        conn._savepoint_stack = []
        conn._savepoint_implicit_begin = False
        conn._has_untracked_savepoint = False

        execute_mock = AsyncMock()
        with patch.object(conn, "execute", new=execute_mock):
            result = await pool._reset_connection(conn)

    execute_mock.assert_not_called()
    assert result is True
