"""Pin: ``DqliteConnection`` raises a *sibling-task-specific*
``InterfaceError`` when task B tries to ``begin_transaction()``
while task A holds an in-flight transaction on the same
connection.

The owner-task arm at connection.py:2470 raises
"Nested transactions are not supported; use SAVEPOINT directly".
The sibling-task arm at connection.py:2473-2474 raises
"Cannot start transaction: connection is in a transaction owned
by another task ... Each task should use its own connection from
the pool."

A regression that collapses the two arms into the owner-task
message would point cross-task users at SAVEPOINT — the wrong
remedy. The fix the original tx-009 introduced disambiguated
the messages; this pin guards against silent regression.
"""

import asyncio
import contextlib

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def _prime_in_transaction(owner_task: asyncio.Task[object]) -> DqliteConnection:
    """Build a DqliteConnection in mid-transaction owned by
    ``owner_task``. Bypasses the wire — we exercise only the
    pre-await sibling-task discriminator."""
    import os

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._closed = False
    conn._in_transaction = True
    conn._tx_owner = owner_task
    conn._has_untracked_savepoint = False
    conn._creator_pid = os.getpid()
    conn._pool_released = False
    conn._bound_loop_ref = None  # lazily bound on first _check_in_use
    conn._in_use = False
    conn._savepoint_stack = []
    return conn


@pytest.mark.asyncio
async def test_sibling_task_begin_raises_separate_connection_message() -> None:
    """The owner is task A; task B's begin_transaction must see
    the sibling-task message, NOT the SAVEPOINT message."""

    async def task_a_holder() -> None:
        # Hold the "owner" identity until the test releases.
        await asyncio.sleep(10)

    task_a = asyncio.create_task(task_a_holder())
    try:
        conn = _prime_in_transaction(owner_task=task_a)

        async def sibling_task() -> None:
            with pytest.raises(InterfaceError) as ei:
                async with conn.transaction():
                    pytest.fail("should not reach")
            msg = str(ei.value)
            # Sibling-task arm:
            assert "owned by another task" in msg
            assert "separate connection" in msg or "own connection" in msg
            # NOT the owner-task arm:
            assert "SAVEPOINT" not in msg

        task_b = asyncio.create_task(sibling_task())
        await task_b
    finally:
        task_a.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_a


@pytest.mark.asyncio
async def test_owner_task_nested_begin_raises_savepoint_message() -> None:
    """Positive control: same task entering begin_transaction while
    already in transaction must see the SAVEPOINT-recommendation
    arm, not the cross-task message. Pins the owner-arm side of
    the discriminator."""
    me = asyncio.current_task()
    assert me is not None
    conn = _prime_in_transaction(owner_task=me)

    with pytest.raises(InterfaceError) as ei:
        async with conn.transaction():
            pytest.fail("should not reach")
    msg = str(ei.value)
    # Owner-task arm:
    assert "Nested transactions" in msg or "SAVEPOINT" in msg
    # NOT the sibling-task arm:
    assert "owned by another task" not in msg
