"""``transaction()`` surfaces a SAVEPOINT-specific diagnostic when an untracked SAVEPOINT is in
flight, instead of letting BEGIN reach the wire and return a confusing nested-transaction error."""

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
    conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
    assert conn._has_untracked_savepoint is True
    assert conn._in_transaction is False

    # The guard must fire before any execute reaches the wire.
    conn.execute = AsyncMock(side_effect=AssertionError("BEGIN must not be issued"))

    with pytest.raises(InterfaceError, match="SAVEPOINT.*auto-begun"):
        async with conn.transaction():
            pytest.fail("body must not run; guard should have fired")
