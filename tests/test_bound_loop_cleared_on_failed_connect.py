"""``DqliteConnection.connect()`` clears ``_bound_loop_ref`` on the
never-connected failure path so a retry from a different event loop
(legitimate in unit-test teardown or ``asyncio.run()`` re-entry)
does not fail ``_check_in_use`` with "bound to a different event
loop" against a ghost binding from the first failed attempt.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


def test_failed_connect_clears_bound_loop() -> None:
    """Simulate a connect-refused on one loop, then run a fresh
    ``asyncio.run`` on a second loop. The second run must succeed
    the pre-connect binding (or at least not fail with the
    "different event loop" ProgrammingError)."""
    # A closed port — asyncio.open_connection will raise
    # ConnectionRefusedError almost immediately.
    conn = DqliteConnection("127.0.0.1:1", timeout=0.5)

    async def _run() -> None:
        with pytest.raises(DqliteConnectionError):
            await conn.connect()

    # First loop: connect fails.
    asyncio.run(_run())
    # Second loop: connect again on the same instance. If
    # ``_bound_loop_ref`` had not been cleared on the failure path,
    # this would raise ProgrammingError("bound to a different event
    # loop") — or under the weakref shape, "bound to a closed event
    # loop" once the first loop was GC'd.
    asyncio.run(_run())
    # After both failures, the binding should still be cleared so a
    # third run can try again.
    assert conn._bound_loop_ref is None
