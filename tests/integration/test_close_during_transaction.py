"""Closing mid-transaction: the server rolls back, and the closed object stays closed."""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection, InterfaceError


@pytest.mark.integration
class TestCloseDuringTransaction:
    async def test_raw_begin_then_close_rolls_back_server_side(self, cluster_address: str) -> None:
        conn1 = DqliteConnection(cluster_address)
        try:
            await conn1.connect()
            await conn1.execute("DROP TABLE IF EXISTS test_close_during_tx")
            await conn1.execute("CREATE TABLE test_close_during_tx (id INTEGER PRIMARY KEY)")
            await conn1.execute("BEGIN")
            await conn1.execute("INSERT INTO test_close_during_tx (id) VALUES (42)")
            assert conn1.in_transaction is True
        finally:
            await conn1.close()
        assert conn1.in_transaction is False
        assert conn1.closed and not conn1.is_connected

        async with DqliteConnection(cluster_address) as conn2:
            assert await conn2.fetchall("SELECT id FROM test_close_during_tx") == []

    async def test_closed_connection_cannot_reconnect(self, cluster_address: str) -> None:
        conn = DqliteConnection(cluster_address)
        await conn.connect()
        await conn.execute("BEGIN")
        await conn.close()
        with pytest.raises(InterfaceError, match="closed"):
            await conn.connect()
        with pytest.raises(InterfaceError, match="closed"):
            await conn.execute("SELECT 1")
        await conn.close()  # idempotent
