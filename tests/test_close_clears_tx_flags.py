"""``DqliteConnection.close()`` clears ``_in_transaction`` and ``_tx_owner``,
mirroring ``_invalidate``, so a reconnect on the same instance does not see
stale tx state and wrongly raise the nested-transaction diagnostic."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestCloseClearsTxFlagsAsync:
    async def test_close_clears_in_transaction_flag(self, conn: DqliteConnection) -> None:
        # Raw BEGIN sets _in_transaction without binding _tx_owner.
        conn._in_transaction = True
        conn._pool_released = False  # avoid the close() early-return branch

        await conn.close()

        assert conn._in_transaction is False

    async def test_close_clears_tx_owner(self, conn: DqliteConnection) -> None:
        # The transaction() context manager binds _tx_owner.
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()
        conn._pool_released = False

        await conn.close()

        assert conn._tx_owner is None

    async def test_close_then_reconnect_does_not_leak_nested_diagnostic(
        self,
    ) -> None:
        """After close + reconnect the flags are clean, so transaction() does not
        misfire the nested-transaction diagnostic (the user-visible bite)."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        # Raw BEGIN leaves _tx_owner None — the path that misclassifies as "Nested".
        conn._tx_owner = None
        conn._pool_released = False

        await conn.close()

        assert conn.in_transaction is False
        assert conn._tx_owner is None

    async def test_close_via_pool_released_shortcut_does_not_touch_flags(
        self,
    ) -> None:
        """The pool-released early-return is a no-op on the flags; the pool's
        _reset_connection / _invalidate owns cleanup for that path."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()
        conn._pool_released = True  # take early-return branch

        await conn.close()

        assert conn._in_transaction is True
        assert conn._tx_owner is not None
