"""The broken-conn cleanup in ``acquire()`` must narrow its suppression:
nested CancelledError and programming errors (AttributeError) propagate,
transport errors (OSError) are caught + DEBUG-logged, and ``_size`` always
decrements. Broad ``suppress(BaseException)`` would silently drop a second
cancel and abandon half-done cleanup."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class _BrokenConn:
    """``DqliteConnection`` stand-in: reports ``is_connected = True`` at acquire
    so the user's body can flip it False to drive the broken-cleanup branch."""

    def __init__(self, close_side_effect: Any = None) -> None:
        self.is_connected = True
        self._pool_released = False
        self._address = "stub:9001"
        self._close_side_effect = close_side_effect
        self.close_calls = 0
        # Mirror the slice of DqliteConnection that _socket_looks_dead walks
        # (protocol -> writer/reader -> transport.is_closing / reader.at_eof)
        # so the pre-ping reports healthy at acquire entry.
        protocol = MagicMock()
        protocol.is_wire_coherent = True
        transport = MagicMock()
        transport.is_closing.return_value = False
        protocol._writer = MagicMock()
        protocol._writer.transport = transport
        protocol._reader = MagicMock()
        protocol._reader.at_eof.return_value = False
        self._protocol = protocol

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_side_effect is None:
            return
        if isinstance(self._close_side_effect, BaseException):
            raise self._close_side_effect
        await self._close_side_effect()


def _make_pool_with_broken_conn(broken: _BrokenConn) -> ConnectionPool:
    """Pool pre-populated with ``broken`` (and ``_size`` bumped) so ``acquire``
    yields it from the queue."""
    cluster = MagicMock(spec=ClusterClient)
    pool = ConnectionPool(addresses=["stub:9001"], cluster=cluster, max_size=1)
    pool._pool.put_nowait(broken)  # type: ignore[arg-type]
    pool._size = 1
    return pool


def _stub_drain_idle(pool: ConnectionPool) -> None:
    async def _noop(*_args: object, **_kwargs: object) -> None:
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
    """TimeoutError (an OSError subclass) is caught via the single OSError
    entry, not a redundant explicit TimeoutError entry."""
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

    async def explode(*_args: object, **_kwargs: object) -> None:
        # OSError is in _POOL_CLEANUP_EXCEPTIONS: absorbed + DEBUG-logged.
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
