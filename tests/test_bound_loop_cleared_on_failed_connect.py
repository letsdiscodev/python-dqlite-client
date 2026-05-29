"""connect() clears _bound_loop_ref on the never-connected failure path so a retry from
a different event loop is not rejected by a ghost binding from the first failed attempt.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


def test_failed_connect_clears_bound_loop() -> None:
    """Connect-refused on one loop, then a fresh asyncio.run on a second loop must not
    fail with a "different event loop" error from the first attempt's ghost binding."""
    conn = DqliteConnection("127.0.0.1:1", timeout=0.5)  # closed port → refused fast

    async def _run() -> None:
        with pytest.raises(DqliteConnectionError):
            await conn.connect()

    asyncio.run(_run())
    asyncio.run(_run())
    assert conn._bound_loop_ref is None
