"""close()'s fork-pid short-circuit nulls _close_done and _closed_event (parent-
loop-bound primitives) so any child-side helper touching them hits the is-None
guard rather than asyncio's "attached to a different loop" diagnostic."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


async def test_pool_close_fork_shortcut_nulls_close_done() -> None:
    pool = ConnectionPool(["localhost:9001"], max_size=1)
    # Simulate the parent's mid-close state, then forge the pid mismatch.
    pool._close_done = asyncio.Event()
    pool._closed_event = asyncio.Event()

    with patch("dqliteclient.connection.get_current_pid", return_value=pool._creator_pid + 1):
        await pool.close()

    assert pool._closed is True
    assert pool._close_done is None, (
        "fork-pid short-circuit must null _close_done so a child-side "
        "helper touching it trips the canonical None-guard rather "
        "than asyncio's loop-mismatch RuntimeError"
    )
    assert pool._closed_event is None, (
        "fork-pid short-circuit must null _closed_event for the same "
        "reason — it is loop-bound to the parent and unsafe to touch "
        "from the child"
    )


async def test_pool_close_fork_shortcut_marks_drain_complete() -> None:
    """The fork short-circuit must mark _drain_complete so the second-caller arm
    does not run _drain_remaining_after_cancel into parent-loop-bound primitives."""
    pool = ConnectionPool(["localhost:9001"], max_size=1)
    pool._close_done = asyncio.Event()
    pool._drain_complete = False

    with patch("dqliteclient.connection.get_current_pid", return_value=pool._creator_pid + 1):
        await pool.close()

    assert pool._drain_complete is True, (
        "fork-pid short-circuit must mark _drain_complete so the "
        "second-caller arm's after-cancel sweep does not fire into "
        "parent-loop-bound code"
    )
