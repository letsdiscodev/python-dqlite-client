"""Pin _invalidate's atomic clearing of _in_transaction / _tx_owner.

External invalidation can land mid-``transaction()`` before its
``finally`` runs; without an atomic clear a stale ``_tx_owner=<dead
task>`` makes ``_check_in_use`` reject the next op with a misleading error.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


def test_invalidate_clears_in_transaction_flag(conn: DqliteConnection) -> None:
    conn._in_transaction = True
    conn._invalidate()
    assert conn._in_transaction is False


def test_invalidate_clears_tx_owner(conn: DqliteConnection) -> None:
    # Production reads `is` for the current-task comparison; any non-None confirms the clear.
    sentinel = object()
    conn._tx_owner = sentinel  # type: ignore[assignment]
    conn._invalidate()
    assert conn._tx_owner is None


class TestInvalidateClearsTxStateAsync:
    async def test_clears_under_external_invalidation_mid_tx(self) -> None:
        """``_invalidate`` mid-tx clears both flags even though no
        ``transaction()`` ``finally`` ever ran."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()

        conn._invalidate()

        assert conn._in_transaction is False
        assert conn._tx_owner is None
        assert conn._in_use is False
        assert conn._protocol is None
