"""Concurrent ``pool.close()`` callers must serialise on the first
caller's drain rather than each racing ``_drain_idle``.

Two awaiters must see exactly one "pool.close: draining" log record
and must both return only after the drain has finished.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _SlowCloseConn:
    def __init__(self, gate: asyncio.Event) -> None:
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
        self._gate = gate
        self.close_called = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        # Block until the test explicitly releases us.
        await self._gate.wait()
        self._protocol = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_concurrent_close_logs_drain_once_and_both_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = asyncio.Event()
    conn = _SlowCloseConn(gate)

    async def _connect(**kwargs: Any) -> _SlowCloseConn:
        return conn

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _connect

    pool = ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )
    # Seed one idle connection so _drain_idle actually does work.
    pool._size = 1
    pool._pool.put_nowait(conn)  # type: ignore[arg-type]

    caplog.set_level(logging.DEBUG, logger="dqliteclient.pool")

    close_a = asyncio.create_task(pool.close())
    close_b = asyncio.create_task(pool.close())

    # Let both tasks start and block on the gated close().
    await asyncio.sleep(0.05)
    assert not close_a.done()
    assert not close_b.done()

    # Release the slow close so _drain_idle can finish.
    gate.set()
    await asyncio.wait_for(asyncio.gather(close_a, close_b), timeout=1.0)

    drain_logs = [r for r in caplog.records if r.message.startswith("pool.close: draining")]
    assert len(drain_logs) == 1, f"expected a single drain log, got {len(drain_logs)}: {drain_logs}"
    # Both callers must only return after close() has actually
    # finished the drain (idempotency contract).
    assert conn.close_called
    assert pool._closed
