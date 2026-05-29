"""Pin: ``_release`` narrows its close() suppress to ``_POOL_CLEANUP_EXCEPTIONS``,
absorbing transport-failure shapes while letting programmer-bug shapes
(AttributeError, TypeError) propagate.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self, *, close_side_effect: BaseException | None = None) -> None:
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol = MagicMock()
        self._protocol.is_wire_coherent = True
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False
        self._close_side_effect = close_side_effect
        self.close_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_side_effect is not None:
            raise self._close_side_effect
        self._pool_released = True
        self._protocol = None  # type: ignore[assignment]


def _make_pool_with_fake_cluster(
    fake_conn: _FakeConn,
    *,
    max_size: int = 1,
) -> ConnectionPool:
    async def _connect(**kwargs: Any) -> _FakeConn:
        return fake_conn

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _connect
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=max_size,
        timeout=1.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_release_absorbs_close_oserror_under_pool_cleanup_exceptions() -> None:
    """The closed-pool branch absorbs an OSError from close(): release returns
    normally and marks the connection released."""
    fake = _FakeConn(close_side_effect=OSError("transport already closed"))
    pool = _make_pool_with_fake_cluster(fake)

    cm = pool.acquire()
    conn = await cm.__aenter__()
    assert conn is fake
    # Close the pool while holding the conn so _release takes the closed arm.
    pool._closed = True
    await cm.__aexit__(None, None, None)

    assert fake.close_calls >= 1, "release path must invoke conn.close()"
    assert fake._pool_released, "release must mark conn._pool_released"


@pytest.mark.asyncio
async def test_release_propagates_attribute_error_from_close() -> None:
    """An AttributeError from close() (a programmer-bug shape) surfaces from
    _release rather than being swallowed."""
    fake = _FakeConn(close_side_effect=AttributeError("refactor mistake"))
    pool = _make_pool_with_fake_cluster(fake)

    cm = pool.acquire()
    await cm.__aenter__()
    pool._closed = True
    with pytest.raises(AttributeError, match="refactor mistake"):
        await cm.__aexit__(None, None, None)

    with contextlib.suppress(Exception):
        await pool.close()


def test_pool_cleanup_exceptions_includes_oserror() -> None:
    """Pin: the narrow tuple covers OSError."""
    from dqliteclient.pool import _POOL_CLEANUP_EXCEPTIONS

    assert OSError in _POOL_CLEANUP_EXCEPTIONS


def test_pool_cleanup_exceptions_does_not_include_attribute_error() -> None:
    """Pin: programmer-bug shapes (AttributeError, TypeError) are NOT in the tuple."""
    from dqliteclient.pool import _POOL_CLEANUP_EXCEPTIONS

    assert AttributeError not in _POOL_CLEANUP_EXCEPTIONS
    assert TypeError not in _POOL_CLEANUP_EXCEPTIONS
