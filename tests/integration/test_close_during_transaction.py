"""Close + reconnect lifecycle for raw transactions, end-to-end.

Two load-bearing contracts the unit suite cannot pin alone:

1. The dqlite server implicitly rolls back any in-flight transaction
   when the client socket closes. A fresh connection MUST see no
   uncommitted rows.

2. After ``close()`` returns on a connection that was holding a raw
   ``BEGIN``, the client-side flags (``in_transaction``, ``_tx_owner``)
   must be clean. Reconnecting the same instance and calling
   ``transaction()`` must NOT raise the misleading "Nested
   transactions" diagnostic. (Load-bearing for the close-clears-flags
   fix.)

Without these tests, regressions in either the client-layer
close-clears-flags discipline or the server's implicit-rollback-on-
disconnect contract would only surface in production. The pool-
release variant of this contract (in-tx connection released to the
pool yields a clean slot on next acquire) lives in
``test_pool_reset_no_tx_preserves_slot.py`` (mock-driven) — the
pool's leader-find loop chases redirect addresses that may not be
reachable from the docker-host test-runner, so an end-to-end pool
variant is gated on cluster-fixture work outside this issue's scope.
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

        # Local flags must be clean post-close (close-clears-flags).
        assert conn1.in_transaction is False
        assert conn1._tx_owner is None

        # Fresh connection: row absent — the server rolled back when
        # the prior socket closed mid-tx.
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

            # Flags clean after close.
            assert conn.in_transaction is False
            assert conn._tx_owner is None

            # Reconnect on the same instance and exercise transaction()
            # — must succeed, not raise "Nested".
            await conn.connect()
            async with conn.transaction():
                await conn.execute("INSERT INTO test_close_reconnect_tx (id) VALUES (1)")

            rows = await conn.fetchall("SELECT id FROM test_close_reconnect_tx")
            assert rows == [[1]]
        finally:
            await conn.close()
