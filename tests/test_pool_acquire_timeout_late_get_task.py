"""acquire()'s capacity-wait timeout demux must use a live get_task.done()
check, not the pre-yield ``in done`` snapshot: a sibling put_nowait can resolve
get_task during the post-wait ``await closed_task`` yield, and the snapshot demux
would discard that conn and permanently leak a capacity slot."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol = MagicMock()
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False
        self.close_called = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        self._protocol = None  # type: ignore[assignment]


def _make_pool() -> ConnectionPool:
    async def _connect(**_: Any) -> _FakeConn:
        return _FakeConn()

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _connect
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=0.1,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_acquire_timeout_race_does_not_discard_late_winning_get_task() -> None:
    """A sibling put_nowait resolves get_task during the post-wait yield; the
    conn must reach the user (or the queue), not be discarded by the stale demux."""
    pool = _make_pool()

    # Pretend max_size is reached so acquire() enters the capacity-wait branch.
    pool._size = 1

    phantom = _FakeConn(name="phantom")
    original_put_nowait = pool._pool.put_nowait

    import dqliteclient.pool as pool_mod

    real_wait = asyncio.wait
    call_count = 0

    async def fake_wait(
        tasks: Any, *, timeout: Any = None, return_when: Any = None
    ) -> tuple[set[Any], set[Any]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Drop phantom into the queue before get_task's first __step runs;
            # the post-wait ``await closed_task`` yield then lets get_task consume
            # it, so the stale ``in done`` snapshot routes it to discard.
            original_put_nowait(phantom)  # type: ignore[arg-type]
            # Timeout shape: done=empty, both tasks still pending.
            return set(), set(tasks)
        return await real_wait(tasks, timeout=timeout, return_when=return_when)

    pool_mod.asyncio.wait = fake_wait  # type: ignore[attr-defined]
    received: object | None = None
    try:
        async with pool.acquire() as conn:
            received = conn
    finally:
        pool_mod.asyncio.wait = real_wait  # type: ignore[attr-defined]

    assert received is phantom, (
        f"acquire returned {received!r}, not the phantom that was put "
        "into the queue during the capacity-wait race — the post-wait "
        "demux's stale 'done' snapshot dropped phantom on the floor"
    )

    # Reset _size so close() does not hit the underflow guard (__aexit__'s
    # _release already decremented via _release_reservation).
    pool._size = 0
    await pool.close()


@pytest.mark.asyncio
async def test_put_back_or_release_late_winner_queuefull_falls_back_to_close() -> None:
    """On QueueFull the late-winner helper must close the conn and decrement
    _size (which wakes sibling acquirers), not leak it."""
    pool = _make_pool()

    # Pre-fill the bounded queue so the helper's put_nowait raises QueueFull.
    pool._size = 1
    occupant = _FakeConn(name="occupant")
    pool._pool.put_nowait(occupant)  # type: ignore[arg-type]
    assert pool._pool.full()

    late_winner = _FakeConn(name="late_winner")
    await pool._put_back_or_release_late_winner(late_winner)  # type: ignore[arg-type]

    assert late_winner.close_called is True
    assert pool._size == 0

    queued = pool._pool.get_nowait()
    assert queued is occupant
    await pool.close()
