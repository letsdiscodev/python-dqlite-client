"""The QueueFull cleanup arm of ``acquire()`` must route its ``_size``
decrement through ``_release_reservation`` so waiters parked on
``closed_event.wait()`` get woken.

Inline ``self._size -= 1`` skips ``_signal_state_change`` — meaning a
sibling acquirer waiting for capacity only wakes via the 10s poll
timeout. Assert the closed event gets set when the cleanup arm fires.
"""

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
        if self._pool_released or self._protocol is None:
            return
        self._protocol = None

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _make_pool() -> tuple[ConnectionPool, list[_FakeConn]]:
    created: list[_FakeConn] = []

    async def _connect(**kwargs: Any) -> _FakeConn:
        c = _FakeConn(name=f"c{len(created)}")
        created.append(c)
        return c

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _connect
    pool = ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )
    return pool, created


@pytest.mark.asyncio
async def test_queuefull_cleanup_wakes_waiters_via_state_signal() -> None:
    """Exercise the cancel-win QueueFull arm and assert the closed
    event fires after cleanup (proof that ``_release_reservation``
    ran, since only it calls ``_signal_state_change``)."""
    pool, created = _make_pool()

    # Occupy the only slot.
    blocking = await pool.acquire().__aenter__()
    assert blocking is created[0]

    # Force-materialise the closed event and clear it so we can
    # observe whether the QueueFull cleanup fires it.
    pool._get_closed_event()
    assert pool._closed_event is not None
    pool._closed_event.clear()

    import dqliteclient.pool as pool_mod

    real_wait = pool_mod.asyncio.wait
    original_put_nowait = pool._pool.put_nowait

    async def fake_wait(tasks, *, timeout=None, return_when):  # type: ignore[no-untyped-def]
        original_put_nowait(blocking)
        await real_wait(tasks, timeout=0.5, return_when=return_when)
        pool._pool.put_nowait = _raise_queue_full  # type: ignore[method-assign]
        raise asyncio.CancelledError

    def _raise_queue_full(_conn: object) -> None:
        raise asyncio.QueueFull

    pool_mod.asyncio.wait = fake_wait  # type: ignore[assignment]
    try:
        cm = pool.acquire()
        try:
            await cm.__aenter__()
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected CancelledError")
    finally:
        pool_mod.asyncio.wait = real_wait  # type: ignore[assignment]
        pool._pool.put_nowait = original_put_nowait  # type: ignore[method-assign]

    # After the cleanup arm runs, the closed event must be set — that
    # is the signal ``_release_reservation`` emits via
    # ``_signal_state_change``. An inline ``self._size -= 1`` would
    # leave the event cleared.
    assert pool._closed_event.is_set(), (
        "QueueFull cleanup must route through _release_reservation "
        "so sibling acquirers parked on closed_event.wait() wake promptly"
    )

    await pool.close()
