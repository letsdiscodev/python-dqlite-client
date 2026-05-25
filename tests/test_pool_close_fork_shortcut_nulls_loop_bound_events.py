"""Pin: ``ConnectionPool.close``'s fork-pid short-circuit nulls
``_close_done`` and ``_closed_event`` (loop-bound primitives
inherited from the parent) so a child-side helper that touches them
trips the canonical ``is None`` guard rather than asyncio's deep
``"got Future <Future pending> attached to a different loop"``
diagnostic.

Pre-fix the short-circuit flipped ``_closed=True`` and returned, but
left ``_close_done`` pointing at the parent loop's ``asyncio.Event``.
The rationale comment at the short-circuit explicitly invokes this
hazard ("awaiting that Event in the child's fresh loop hangs
forever"); the implementation defended only the immediate close()
return path, leaving any other code path that touches the inherited
Event exposed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


async def test_pool_close_fork_shortcut_nulls_close_done() -> None:
    pool = ConnectionPool(["localhost:9001"], max_size=1)
    # Simulate the parent's mid-close state: assign a parent-loop-
    # bound Event then run the fork short-circuit by forging the pid
    # mismatch.
    pool._close_done = asyncio.Event()
    pool._closed_event = asyncio.Event()

    with patch("dqliteclient.connection.get_current_pid", return_value=pool._creator_pid + 1):
        await pool.close()

    # The fork short-circuit fired (we observe its effects on the
    # closed flags) and the loop-bound primitives are now None so
    # any subsequent touch hits the documented None-guard.
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
    """The second-caller arm at the ``if self._closed:`` early-return
    runs ``_drain_remaining_after_cancel`` when ``_drain_complete`` is
    False. That sweep touches parent-loop-bound queue primitives, so
    the fork short-circuit must mark ``_drain_complete`` so the
    second-caller arm short-circuits cleanly.
    """
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
