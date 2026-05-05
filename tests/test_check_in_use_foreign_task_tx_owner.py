"""``_check_in_use``'s "owned by another task" arm raises
``InterfaceError`` when a sibling task tries to use the connection
while another task holds the transaction. Pin the diagnostic
shape so a regression cannot silently weaken the concurrency
guard.
"""

import asyncio
import contextlib

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_foreign_task_during_active_tx_raises_interface_error() -> None:
    """Task A holds a tx; task B uses the connection. B must surface
    InterfaceError citing both task identities."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    # Mark the connection as in a tx owned by a synthetic task.
    conn._in_transaction = True
    fake_owner = asyncio.create_task(asyncio.sleep(60))
    conn._tx_owner = fake_owner

    try:
        with pytest.raises(InterfaceError) as excinfo:
            conn._check_in_use()
        msg = str(excinfo.value)
        assert "owned by another task" in msg
        # Both task reprs should appear so an operator can correlate.
        assert "owner" in msg
        assert "current" in msg
    finally:
        fake_owner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fake_owner


@pytest.mark.asyncio
async def test_same_task_during_active_tx_passes() -> None:
    """The owner task itself is allowed to continue using the
    connection — ``_check_in_use`` only rejects foreign tasks."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    # Should not raise.
    conn._check_in_use()
