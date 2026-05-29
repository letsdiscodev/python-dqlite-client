"""close() sets _closed_event from finally (like _close_done.set()): an exception
between _closed=True and the event-set would otherwise leave parked acquirers on
_closed_event.wait() until their own acquire timeout fires."""

from __future__ import annotations

import asyncio
import os

import pytest

from dqliteclient import pool as pool_mod
from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


def _make_pool() -> ConnectionPool:
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._pool = asyncio.Queue(maxsize=1)
    pool._size = 0
    pool._lock = asyncio.Lock()
    pool._closed = False
    pool._closed_flag = [False]
    pool._closed_event = asyncio.Event()
    pool._close_done = None
    pool._drain_complete = False
    pool._finalizer = None
    pool._creator_pid = os.getpid()
    pool._close_timeout = 0.05
    pool._max_size = 1
    pool._cluster = None  # type: ignore[assignment]
    pool._timeout = 1.0
    pool._addresses = []
    return pool


async def test_closed_event_is_set_when_drain_path_raises_before_explicit_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _make_pool()
    captured_event = pool._closed_event
    assert captured_event is not None

    class _SyntheticRaise(Exception):
        """Stand-in for a signal-delivery raise inside the close window."""

    def _raise_during_drain_window(*_args: object, **_kwargs: object) -> None:
        raise _SyntheticRaise("synthetic raise between _closed=True and event-set")

    # logger.debug sits in the drain try arm between _closed=True and the
    # event-set; raising here reproduces the gap the fix closes.
    monkeypatch.setattr(pool_mod.logger, "debug", _raise_during_drain_window)

    with pytest.raises(_SyntheticRaise):
        await pool.close()

    assert captured_event.is_set(), (
        "_closed_event must be set even when the drain-and-publish "
        "try arm raises; the set belongs in finally so parked "
        "acquirers wake without waiting for their acquire timeout"
    )
