"""``DqliteConnection._run_protocol``'s CancelledError arm invalidates
the connection (FIN to the server, transport closed) but does NOT
send INTERRUPT on a fresh socket.

The dqlite C server's ``handle_interrupt`` (gateway.c:951-967, dispatch
at 1373-1383) is keyed on the cancelled connection's ``g->req`` slot.
A fresh socket has ``g->req == NULL``; the server's interrupt handler
returns SUCCESS_V0 without aborting the in-flight query — pure wasted
dial + handshake. Sending INTERRUPT on the cancelled socket would
require holding the cancelled task's ``op_lock`` past cancel
propagation, which defeats the cancel.

The dqlite gateway tears down the per-connection slot when the
cancelled connection's FIN reaches it on the next write attempt.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_run_protocol_invalidates_on_cancel() -> None:
    """When ``_run_protocol`` catches CancelledError, it invalidates
    the connection (writer closed, _protocol cleared) and re-raises.
    No background task is scheduled."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)

    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 7
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 7))

    async def cancelled_op(_p: object, _db: int) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await conn._run_protocol(cancelled_op)

    assert conn._protocol is None
    assert conn._invalidation_cause is not None


@pytest.mark.asyncio
async def test_run_protocol_no_background_task_scheduled_on_cancel() -> None:
    """Cancellation must not leave any orphan task in the running
    loop. If a future regression re-introduces a fire-and-forget
    interrupt task, this test catches it."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 7
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 7))

    async def cancelled_op(_p: object, _db: int) -> None:
        raise asyncio.CancelledError()

    pre_tasks = {id(t) for t in asyncio.all_tasks()}

    with pytest.raises(asyncio.CancelledError):
        await conn._run_protocol(cancelled_op)

    # Drain any scheduled callbacks.
    await asyncio.sleep(0)

    # Any task created after the snapshot would be visible here. Only
    # the running test's own task should still be present.
    new_tasks = {id(t) for t in asyncio.all_tasks() if not t.done()} - pre_tasks
    new_tasks.discard(id(asyncio.current_task()))
    assert not new_tasks, f"unexpected background tasks: {new_tasks}"
