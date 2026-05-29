"""``_run_protocol``'s CancelledError arm invalidates the connection but
sends no INTERRUPT: the server's ``handle_interrupt`` (gateway.c:951-967)
is keyed on the cancelled socket's ``g->req``, so a fresh socket's
INTERRUPT is a no-op and the gateway tears the slot down on the FIN.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_run_protocol_invalidates_on_cancel() -> None:
    """CancelledError invalidates the connection and re-raises; no task scheduled."""
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
    """Cancellation must not leave an orphan fire-and-forget interrupt task."""
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

    new_tasks = {id(t) for t in asyncio.all_tasks() if not t.done()} - pre_tasks
    new_tasks.discard(id(asyncio.current_task()))
    assert not new_tasks, f"unexpected background tasks: {new_tasks}"
