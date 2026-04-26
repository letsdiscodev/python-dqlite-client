"""``transaction()`` ctxmgr surfaces a dedicated diagnostic when an
untracked SAVEPOINT is in flight.

Without the guard, entering ``transaction()`` after issuing
``SAVEPOINT "Foo"`` (parser-rejected name → autobegun server-side tx
with ``_in_transaction=False``, ``_has_untracked_savepoint=True``)
would proceed to send ``BEGIN`` over the wire. The server replies with
``cannot start a transaction within a transaction`` — confusing, since
the user's diagnostic should name the SAVEPOINT root cause.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.fixture
def conn() -> DqliteConnection:
    c = DqliteConnection("localhost:9001")
    c._db_id = 1
    c._protocol = object()  # type: ignore[assignment]
    return c


@pytest.mark.asyncio
async def test_transaction_ctxmgr_rejects_after_untracked_savepoint(
    conn: DqliteConnection,
) -> None:
    # Drive the tracker to: _in_transaction=False, _has_untracked_savepoint=True.
    conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
    assert conn._has_untracked_savepoint is True
    assert conn._in_transaction is False

    # No execute should reach the wire — the guard fires first.
    conn.execute = AsyncMock(side_effect=AssertionError("BEGIN must not be issued"))

    with pytest.raises(InterfaceError, match="SAVEPOINT.*auto-begun"):
        async with conn.transaction():
            pytest.fail("body must not run; guard should have fired")
