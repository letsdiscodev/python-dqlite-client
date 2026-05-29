"""Close + reconnect lifecycle for raw transactions, end-to-end.

Pins two contracts the unit suite cannot: (1) the server implicitly rolls back an
in-flight tx when the socket closes, and (2) close() clears the client-side flags so a
reconnect + transaction() does not raise the misleading "Nested transactions" diagnostic.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection


@pytest.mark.integration
class TestCloseDuringTransaction:
    async def test_raw_begin_then_close_rolls_back_server_side(self, cluster_address: str) -> None:
        """Raw BEGIN + INSERT + close (no COMMIT/ROLLBACK) — fresh
        connection sees the row absent (server-side implicit rollback)."""
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
        assert conn1._tx_owner is None

        # Fresh connection: row absent (server rolled back on the mid-tx socket close).
        conn2 = DqliteConnection(cluster_address)
        try:
            await conn2.connect()
            rows = await conn2.fetchall("SELECT id FROM test_close_during_tx")
            assert rows == []
        finally:
            await conn2.close()

    async def test_close_then_reconnect_does_not_raise_nested_tx(
        self, cluster_address: str
    ) -> None:
        """Same instance: BEGIN + close + reconnect + transaction() —
        the second transaction() must NOT raise "Nested transactions
        are not supported" (the bug shape close-clears-flags fixed)."""
        conn = DqliteConnection(cluster_address)
        try:
            await conn.connect()
            await conn.execute("DROP TABLE IF EXISTS test_close_reconnect_tx")
            await conn.execute("CREATE TABLE test_close_reconnect_tx (id INTEGER PRIMARY KEY)")
            await conn.execute("BEGIN")
            await conn.close()

            assert conn.in_transaction is False
            assert conn._tx_owner is None

            # Reconnect same instance + transaction() must succeed, not raise "Nested".
            await conn.connect()
            async with conn.transaction():
                await conn.execute("INSERT INTO test_close_reconnect_tx (id) VALUES (1)")

            rows = await conn.fetchall("SELECT id FROM test_close_reconnect_tx")
            assert rows == [[1]]
        finally:
            await conn.close()
