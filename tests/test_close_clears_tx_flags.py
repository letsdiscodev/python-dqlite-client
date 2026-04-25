"""Pin ``DqliteConnection.close()`` clearing ``_in_transaction`` and
``_tx_owner``.

``_invalidate`` (the failure path) atomically clears both flags so a
subsequent caller does not see stale transaction state. The intentional
``close()`` path historically forgot to mirror the discipline. After a
raw ``BEGIN`` followed by an explicit ``close()`` and a reconnect on
the same ``DqliteConnection`` instance, the ``_in_transaction`` flag
would remain ``True`` and the next caller would see ``in_transaction``
lie about server-side state — and ``transaction()`` would raise the
misleading "Nested transactions are not supported" diagnostic.

Pin the symmetric invariant: ``close()`` clears both flags before
returning, mirroring ``_invalidate``.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestCloseClearsTxFlagsAsync:
    async def test_close_clears_in_transaction_flag(self, conn: DqliteConnection) -> None:
        # Simulate a raw BEGIN that set _in_transaction without binding
        # _tx_owner (matches _update_tx_flags_from_sql for raw BEGIN).
        conn._in_transaction = True
        # Mark _pool_released=False explicitly so close() does not take
        # the early-return branch.
        conn._pool_released = False

        await conn.close()

        assert conn._in_transaction is False

    async def test_close_clears_tx_owner(self, conn: DqliteConnection) -> None:
        # transaction() context manager binds _tx_owner; simulate that
        # state without actually awaiting BEGIN.
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()
        conn._pool_released = False

        await conn.close()

        assert conn._tx_owner is None

    async def test_close_then_reconnect_does_not_leak_nested_diagnostic(
        self,
    ) -> None:
        """After close + reconnect on the same DqliteConnection, the
        ``transaction()`` context manager must not raise "Nested" — the
        flags from the prior session must be cleared.

        This is the user-visible bite of the bug: without the close()
        clear, in_transaction lies and transaction() raises the wrong
        diagnostic. The test exercises the in-memory state contract
        (no live cluster needed): after close(), assert flags are
        clean so a hypothetical reconnect does not trip
        ``_check_in_use`` / the nested-transaction guard.
        """
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        # raw BEGIN leaves _tx_owner None; this is the path that
        # currently misclassifies as "Nested" because
        # current_task() is also None outside an async context inside
        # the transaction() context manager guard.
        conn._tx_owner = None
        conn._pool_released = False

        await conn.close()

        assert conn.in_transaction is False
        assert conn._tx_owner is None

    async def test_close_via_pool_released_shortcut_does_not_touch_flags(
        self,
    ) -> None:
        """Pool-released connections take an early-return at close()
        because the pool path ran the close already. Flag clearing
        happens via the pool's _reset_connection (or via this issue's
        canonical close clear once that path is reached). The early
        return must not rely on or be confused by stale flag state.

        This test pins that the early-return continues to be a no-op
        on the flags — the pool owns the cleanup for that path."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()
        conn._pool_released = True  # take early-return branch

        await conn.close()

        # Pool-released early-return does not touch flags; the pool's
        # _reset_connection / _invalidate is responsible for that path.
        assert conn._in_transaction is True
        assert conn._tx_owner is not None
