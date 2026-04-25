"""Pin _invalidate's atomic clearing of _in_transaction / _tx_owner.

External invalidation paths (heartbeat, ``call_soon_threadsafe``,
KeyboardInterrupt mid-yield) can land on a connection that is currently
inside a ``transaction()`` context — the ``transaction()`` ``finally``
clause that normally clears the flags may not run before the connection
is observed by another caller (typically the pool). Without an atomic
clear in ``_invalidate``, a stale ``_in_transaction=True, _tx_owner=
<dead task>`` slips out and ``_check_in_use`` rejects the next
operation with a misleading "owned by another task" InterfaceError.

Pin the canonical invariant: after ``_invalidate``, both flags MUST be
cleared regardless of how the prior transaction state ended.
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
    # Use a sentinel object — the production code reads `is` for the
    # current-task comparison. Anything non-None confirms the clear.
    sentinel = object()
    conn._tx_owner = sentinel  # type: ignore[assignment]
    conn._invalidate()
    assert conn._tx_owner is None


class TestInvalidateClearsTxStateAsync:
    async def test_clears_under_external_invalidation_mid_tx(self) -> None:
        """Simulate: ``_invalidate`` lands on a connection that already
        has ``_in_transaction=True`` and ``_tx_owner`` pointing at the
        current task. After invalidation both flags must be cleared,
        even though no ``transaction()`` ``finally`` ever ran."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()

        conn._invalidate()

        assert conn._in_transaction is False
        assert conn._tx_owner is None
        assert conn._in_use is False
        assert conn._protocol is None
