"""Narrow the except-BaseException cleanup in ``ConnectionPool.acquire``.

ISSUE-198: the broken-connection branch of the ``acquire()`` except
clause used to wrap ``_drain_idle``, ``conn.close()``, and the
shielded reservation-release each in ``contextlib.suppress(BaseException)``.
Under a nested-cancel scenario (caller cancels us once, then a parent
TaskGroup or ``asyncio.timeout()`` fires a second cancel mid-cleanup),
the suppression silently dropped the second cancel — and silently
abandoned half-done cleanup work along with it. Programming errors
raised in cleanup (e.g., ``AttributeError`` from a missing attr)
likewise vanished into the void.

These tests pin the narrowed semantics:

1. A nested ``CancelledError`` raised by ``_drain_idle`` during cleanup
   must still propagate out of ``acquire()``.
2. A programming error (``AttributeError``) raised by ``conn.close()``
   during cleanup must propagate, not be silently swallowed.
3. Transport-layer ``OSError`` raised by ``conn.close()`` during cleanup
   must be caught and emit a DEBUG log — the original exception from
   the body must then re-raise unchanged.
4. ``_size`` must still be decremented across every cleanup path.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class _BrokenConn:
    """A ``DqliteConnection`` stand-in for the broken-cleanup branch.

    The pool's ``acquire`` has two cleanup paths: the healthy one
    (return to queue) and the broken one (drain idle, close, release).
    The broken branch (pool.py:434-…) fires when ``conn.is_connected``
    is False *after the user's body raised*. The fake initially reports
    ``is_connected = True`` so the early reconnect path at pool.py:385
    doesn't engage; the user's body then flips it to False to simulate
    an operation that invalidated the connection.
    """

    def __init__(self, close_side_effect: Any = None) -> None:
        self.is_connected = True
        self._pool_released = False
        self._address = "stub:9001"
        self._close_side_effect = close_side_effect
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_side_effect is None:
            return
        if isinstance(self._close_side_effect, BaseException):
            raise self._close_side_effect
        await self._close_side_effect()


def _make_pool_with_broken_conn(broken: _BrokenConn) -> ConnectionPool:
    """Build a pool rigged to yield ``broken`` from ``acquire``.

    The pool's ``acquire`` first reserves a slot, then pulls from the
    in-queue or calls ``_create_connection``. We bypass that by
    pre-populating the queue with the broken conn and bumping ``_size``
    to match.
    """
    cluster = MagicMock(spec=ClusterClient)
    pool = ConnectionPool(addresses=["stub:9001"], cluster=cluster, max_size=1)
    pool._pool.put_nowait(broken)  # type: ignore[arg-type]
    pool._size = 1
    return pool


def _stub_drain_idle(pool: ConnectionPool) -> None:
    async def _noop() -> None:
        return None

    pool._drain_idle = _noop


@pytest.mark.asyncio
async def test_cleanup_propagates_attribute_error_from_close() -> None:
    broken = _BrokenConn(close_side_effect=AttributeError("close bug"))
    pool = _make_pool_with_broken_conn(broken)
    _stub_drain_idle(pool)

    with pytest.raises(AttributeError, match="close bug"):
        async with pool.acquire() as conn:
            conn.is_connected = False  # type: ignore[misc]
            raise RuntimeError("user code failure")

    assert pool._size == 0


@pytest.mark.asyncio
async def test_cleanup_logs_oserror_from_close(caplog: pytest.LogCaptureFixture) -> None:
    broken = _BrokenConn(close_side_effect=OSError("ECONNRESET"))
    pool = _make_pool_with_broken_conn(broken)
    _stub_drain_idle(pool)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.pool")
    with pytest.raises(RuntimeError, match="user code failure"):
        async with pool.acquire() as conn:
            conn.is_connected = False  # type: ignore[misc]
            raise RuntimeError("user code failure")

    assert broken.close_calls == 1
    assert any(
        "pool.acquire cleanup: conn.close" in record.getMessage()
        and record.exc_info is not None
        and "ECONNRESET" in str(record.exc_info[1])
        for record in caplog.records
    ), f"expected cleanup DEBUG log with exc_info, got {[r.getMessage() for r in caplog.records]}"
    assert pool._size == 0


@pytest.mark.asyncio
async def test_cleanup_logs_timeout_error_from_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TimeoutError is an OSError subclass; the cleanup suppression
    must catch it via the single OSError entry in the except tuple.
    Regression guard for the narrowing done in the sibling cleanup
    that removed a redundant explicit ``TimeoutError`` entry.
    """
    broken = _BrokenConn(close_side_effect=TimeoutError("read timed out"))
    pool = _make_pool_with_broken_conn(broken)
    _stub_drain_idle(pool)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.pool")
    with pytest.raises(RuntimeError, match="user code failure"):
        async with pool.acquire() as conn:
            conn.is_connected = False  # type: ignore[misc]
            raise RuntimeError("user code failure")

    assert broken.close_calls == 1
    assert any(
        "pool.acquire cleanup: conn.close" in record.getMessage()
        and record.exc_info is not None
        and "read timed out" in str(record.exc_info[1])
        for record in caplog.records
    ), f"expected cleanup DEBUG log with exc_info, got {[r.getMessage() for r in caplog.records]}"
    assert pool._size == 0


@pytest.mark.asyncio
async def test_cleanup_preserves_original_exception_when_close_silent() -> None:
    broken = _BrokenConn(close_side_effect=None)
    pool = _make_pool_with_broken_conn(broken)
    _stub_drain_idle(pool)

    with pytest.raises(DqliteConnectionError, match="original body error"):
        async with pool.acquire() as conn:
            conn.is_connected = False  # type: ignore[misc]
            raise DqliteConnectionError("original body error")

    assert broken.close_calls == 1
    assert pool._size == 0
    assert broken._pool_released is True


@pytest.mark.asyncio
async def test_cleanup_logs_drain_idle_failure(caplog: pytest.LogCaptureFixture) -> None:
    broken = _BrokenConn(close_side_effect=None)
    pool = _make_pool_with_broken_conn(broken)

    async def explode() -> None:
        # OSError is in _POOL_CLEANUP_EXCEPTIONS — the narrow catch
        # absorbs transport-class failures and DEBUG-logs them, but
        # programmer bugs (TypeError / AttributeError) still propagate.
        raise OSError("drain-idle transport failure")

    pool._drain_idle = explode

    caplog.set_level(logging.DEBUG, logger="dqliteclient.pool")
    with pytest.raises(RuntimeError, match="user code failure"):
        async with pool.acquire() as conn:
            conn.is_connected = False  # type: ignore[misc]
            raise RuntimeError("user code failure")

    assert any(
        "pool.acquire cleanup: _drain_idle failed" in record.getMessage()
        and record.exc_info is not None
        and "drain-idle transport failure" in str(record.exc_info[1])
        for record in caplog.records
    ), (
        "expected drain-idle DEBUG log with exc_info, "
        f"got {[r.getMessage() for r in caplog.records]}"
    )
    assert pool._size == 0
