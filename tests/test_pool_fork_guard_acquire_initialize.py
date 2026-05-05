"""Pin: ``ConnectionPool.initialize`` and
``ConnectionPool.acquire`` raise ``InterfaceError`` when called
after fork (creator pid != current pid).

ISSUE-844 introduced these guards; ISSUE-942 pinned the close()
fork branch only. The initialize / acquire arms were uncovered —
a regression that drops either guard would let a forked child
silently share the parent's TCP fds with corrupted reads / mixed
request-response framing.

Use the same monkeypatch pattern as ISSUE-1093 / ISSUE-966 — no
real fork() needed.
"""

import os

import pytest

from dqliteclient import connection as _conn_mod
from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_initialize_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=1, max_size=2)
    monkeypatch.setattr(_conn_mod, "_current_pid", os.getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_pool_acquire_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    monkeypatch.setattr(_conn_mod, "_current_pid", os.getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        async with pool.acquire():
            pytest.fail("should not reach")
