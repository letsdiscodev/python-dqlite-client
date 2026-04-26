"""_release's three close-during-release branches must share the same
narrow suppression discipline: absorb _POOL_CLEANUP_EXCEPTIONS, let
CancelledError / KeyboardInterrupt / programmer bugs propagate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import OperationalError
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self, close_raises: BaseException | None = None) -> None:
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
        self._close_raises = close_raises
        self.close_called = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        if self._close_raises is not None:
            raise self._close_raises

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _make_pool() -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)

    async def _connect(**kwargs: Any) -> _FakeConn:
        return _FakeConn()

    cluster.connect = _connect
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=2,
        timeout=1.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_release_closed_branch_absorbs_cleanup_exception() -> None:
    pool = _make_pool()
    pool._closed = True
    conn = _FakeConn(close_raises=BrokenPipeError("EPIPE"))
    pool._size = 1
    # Branch 1: pool is closed.
    await pool._release(conn)  # type: ignore[arg-type]
    assert conn.close_called
    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_release_reset_fail_branch_absorbs_operational_error() -> None:
    pool = _make_pool()
    conn = _FakeConn(close_raises=OperationalError(0, "synthetic"))
    pool._size = 1

    async def _reset_fail(c: Any) -> bool:
        return False

    pool._reset_connection = _reset_fail  # type: ignore[assignment]

    # Branch 2: reset returned False.
    await pool._release(conn)  # type: ignore[arg-type]
    assert conn.close_called
    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_release_queuefull_branch_absorbs_cleanup_exception() -> None:
    pool = _make_pool()
    conn = _FakeConn(close_raises=BrokenPipeError("EPIPE"))
    pool._size = 3  # over capacity so put_nowait raises

    # Force the queue full by filling it with placeholders.
    filler = _FakeConn()
    filler2 = _FakeConn()
    pool._pool.put_nowait(filler)  # type: ignore[arg-type]
    pool._pool.put_nowait(filler2)  # type: ignore[arg-type]

    async def _reset_ok(c: Any) -> bool:
        return True

    pool._reset_connection = _reset_ok  # type: ignore[assignment]

    # Branch 3: queue full on put_nowait.
    await pool._release(conn)  # type: ignore[arg-type]
    assert conn.close_called
    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_release_closed_branch_propagates_programmer_bug() -> None:
    """TypeError from close() is a programmer bug and must NOT be
    absorbed — the narrow tuple excludes arbitrary Exceptions."""
    pool = _make_pool()
    pool._closed = True
    conn = _FakeConn(close_raises=TypeError("whoops"))
    pool._size = 1
    with pytest.raises(TypeError, match="whoops"):
        await pool._release(conn)  # type: ignore[arg-type]
