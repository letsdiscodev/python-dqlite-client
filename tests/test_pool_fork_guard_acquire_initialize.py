"""Pin: ``ConnectionPool.initialize`` and
``ConnectionPool.acquire`` raise ``InterfaceError`` when called
after fork (creator pid != current pid).

The fork guards on ``close`` are pinned by a sibling test; this
file covers the ``initialize`` and ``acquire`` arms so a regression
that drops either guard would let a forked child silently share the
parent's TCP fds with corrupted reads / mixed request-response
framing.

Uses ``monkeypatch.setattr`` to spoof ``os.getpid()`` so the test
exercises the post-fork branch without a real ``fork()``.
"""

import os

import pytest

from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_initialize_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=1, max_size=2)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_pool_acquire_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        async with pool.acquire():
            pytest.fail("should not reach")
