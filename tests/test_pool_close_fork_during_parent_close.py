"""close()'s fork short-circuit must run BEFORE the ``if self._closed:`` arm:
otherwise a child whose parent forked mid-close awaits _close_done, an Event
bound to the parent's defunct loop, blocking forever in the child's fresh loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


def _make_pool() -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_close_with_pid_mismatch_returns_even_when_closed_with_pending_event() -> None:
    """close() in a forked child must short-circuit and return even when
    _closed=True and _close_done is set-but-not-completed (parent's _drain_idle
    in flight at fork time); else the child awaits _close_done forever."""
    pool = _make_pool()
    pool._closed = True
    # Inherited Event nobody will set(): a hang surfaces as TimeoutError.
    pool._close_done = asyncio.Event()
    fake_parent_pid = pool._creator_pid + 1
    pool._creator_pid = fake_parent_pid  # Simulate "we are the child."

    with patch("dqliteclient.connection.os.getpid", return_value=fake_parent_pid + 1):
        await asyncio.wait_for(pool.close(), timeout=1.0)

    assert pool._closed is True


@pytest.mark.asyncio
async def test_close_with_pid_mismatch_returns_even_when_closed_with_no_event() -> None:
    """Sibling case: _closed=True but _close_done was never published; child must
    still short-circuit without entering the ``if self._closed:`` arm."""
    pool = _make_pool()
    pool._closed = True
    pool._close_done = None
    fake_parent_pid = pool._creator_pid + 1
    pool._creator_pid = fake_parent_pid

    with patch("dqliteclient.connection.os.getpid", return_value=fake_parent_pid + 1):
        await asyncio.wait_for(pool.close(), timeout=1.0)

    assert pool._closed is True
