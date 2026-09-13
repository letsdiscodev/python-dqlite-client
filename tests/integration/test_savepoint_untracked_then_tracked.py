"""``in_transaction`` is a conservative local flag: SAVEPOINT sets it, RELEASE leaves it,
COMMIT / ROLLBACK clear it, and the server's "no transaction" reply to a redundant
ROLLBACK is the only cost of over-reporting."""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection, InterfaceError, OperationalError
from dqliteclient.connection import is_no_transaction_error


@pytest.mark.integration
async def test_savepoint_flags_are_conservative(cluster_address: str) -> None:
    async with DqliteConnection(cluster_address) as conn:
        assert conn.in_transaction is False
        await conn.execute('SAVEPOINT "Foo"')
        assert conn.in_transaction is True
        await conn.execute("SAVEPOINT inner")
        await conn.execute("RELEASE inner")
        assert conn.in_transaction is True
        await conn.execute('RELEASE "Foo"')
        # The engine is back in autocommit; the flag stays set until an explicit end.
        assert conn.in_transaction is True
        with pytest.raises(OperationalError) as info:
            await conn.execute("ROLLBACK")
        assert is_no_transaction_error(info.value)
        assert conn.in_transaction is False
        assert conn.is_connected


@pytest.mark.integration
async def test_transaction_ctxmgr_after_savepoint_cleanup(cluster_address: str) -> None:
    async with DqliteConnection(cluster_address) as conn:
        await conn.execute("DROP TABLE IF EXISTS test_sp_then_tx")
        await conn.execute("CREATE TABLE test_sp_then_tx (id INTEGER PRIMARY KEY)")
        await conn.execute('SAVEPOINT "Bar"')
        with pytest.raises(InterfaceError, match="Nested transactions"):
            async with conn.transaction():
                pass
        await conn.execute("ROLLBACK")
        assert conn.in_transaction is False
        async with conn.transaction():
            await conn.execute("INSERT INTO test_sp_then_tx (id) VALUES (1)")
        assert await conn.fetchall("SELECT id FROM test_sp_then_tx") == [[1]]
