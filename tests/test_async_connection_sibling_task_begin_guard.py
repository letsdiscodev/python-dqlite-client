"""begin_transaction from a sibling task raises a sibling-specific InterfaceError.

The owner-task arm recommends SAVEPOINT; collapsing the two arms would point cross-task
users at SAVEPOINT, the wrong remedy.
"""

import asyncio
import contextlib

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def _prime_in_transaction(owner_task: asyncio.Task[object]) -> DqliteConnection:
    """Build a mid-transaction DqliteConnection owned by owner_task, bypassing the wire."""
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
    """Task B's begin_transaction sees the sibling-task message, not SAVEPOINT."""

    async def task_a_holder() -> None:
        await asyncio.sleep(10)

    task_a = asyncio.create_task(task_a_holder())
    try:
        conn = _prime_in_transaction(owner_task=task_a)

        async def sibling_task() -> None:
            with pytest.raises(InterfaceError) as ei:
                async with conn.transaction():
                    pytest.fail("should not reach")
            msg = str(ei.value)
            assert "owned by another task" in msg
            assert "separate connection" in msg or "own connection" in msg
            assert "SAVEPOINT" not in msg

        task_b = asyncio.create_task(sibling_task())
        await task_b
    finally:
        task_a.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_a


@pytest.mark.asyncio
async def test_owner_task_nested_begin_raises_savepoint_message() -> None:
    """Same task re-entering begin_transaction sees the SAVEPOINT arm, not cross-task."""
    me = asyncio.current_task()
    assert me is not None
    conn = _prime_in_transaction(owner_task=me)

    with pytest.raises(InterfaceError) as ei:
        async with conn.transaction():
            pytest.fail("should not reach")
    msg = str(ei.value)
    assert "Nested transactions" in msg or "SAVEPOINT" in msg
    assert "owned by another task" not in msg
