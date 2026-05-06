"""Pin: the pool's ``_release`` path narrows
``contextlib.suppress(Exception)`` to ``_POOL_CLEANUP_EXCEPTIONS``,
absorbing legitimate transport-failure shapes from ``conn.close()``
without crashing release while letting programmer-bug shapes
(AttributeError, TypeError) propagate.

The previous version of this test imported ``logger`` and
``_POOL_CLEANUP_EXCEPTIONS`` from the production module and inlined
a duplicated `if isinstance(exc, _POOL_CLEANUP_EXCEPTIONS):
logger.debug(...)` block — it asserted that its OWN inline call
emitted the DEBUG record, never invoking the production code under
test. A regression that widened the suppress back to bare
``Exception``, or one that dropped OSError from the cleanup tuple,
would not have been observable.

This rewrite drives the actual ``pool.acquire()`` /
``ConnectionPool._release`` release path:

  * The closed-pool branch absorbs an OSError from ``conn.close()``
    via ``contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS)`` — release
    completes without re-raising, and ``conn._pool_released`` is set.
  * A programmer-bug shape (AttributeError) is NOT in the tuple — the
    same release path propagates it, surfacing the refactor mistake.

The two constant-shape pins (``includes_oserror`` and
``does_not_include_attribute_error``) remain — they pin the tuple
contents, distinct from the behavioural pin above.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    """Fake connection that tracks close() calls and can raise on close."""

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
    """An OSError from ``conn.close()`` during the closed-pool release
    branch is absorbed by ``contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS)``
    in ``ConnectionPool._release``. The release returns normally and
    marks the connection released; the OSError does not propagate."""
    fake = _FakeConn(close_side_effect=OSError("transport already closed"))
    pool = _make_pool_with_fake_cluster(fake)

    cm = pool.acquire()
    conn = await cm.__aenter__()
    assert conn is fake
    # Close the pool while we hold the connection so ``_release``
    # takes the ``if self._closed:`` arm at line 1501 (or its
    # post-reset re-check at 1520) — both call ``conn.close()``
    # under the cleanup-class suppress.
    pool._closed = True
    # Must NOT raise — the suppress catches OSError from close().
    await cm.__aexit__(None, None, None)

    assert fake.close_calls >= 1, "release path must invoke conn.close()"
    assert fake._pool_released, "release must mark conn._pool_released"


@pytest.mark.asyncio
async def test_release_propagates_attribute_error_from_close() -> None:
    """``contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS)`` must NOT
    absorb programmer-bug shapes. Inject an AttributeError from
    ``conn.close()`` and confirm it surfaces from ``_release`` (rather
    than being silently swallowed as a bare ``except Exception:``
    suppress would have done)."""
    fake = _FakeConn(close_side_effect=AttributeError("refactor mistake"))
    pool = _make_pool_with_fake_cluster(fake)

    cm = pool.acquire()
    await cm.__aenter__()
    pool._closed = True
    with pytest.raises(AttributeError, match="refactor mistake"):
        await cm.__aexit__(None, None, None)

    # Best-effort cleanup so the test fixture doesn't leak the pool.
    with contextlib.suppress(Exception):
        await pool.close()


def test_pool_cleanup_exceptions_includes_oserror() -> None:
    """Pin: the narrow tuple covers OSError. A regression that drops
    OSError from the tuple would convert the suppress branch into a
    propagating exception."""
    from dqliteclient.pool import _POOL_CLEANUP_EXCEPTIONS

    assert OSError in _POOL_CLEANUP_EXCEPTIONS


def test_pool_cleanup_exceptions_does_not_include_attribute_error() -> None:
    """Pin: programmer-bug shapes (AttributeError, TypeError) are NOT
    in the narrow tuple. The release path propagates them, surfacing
    refactor mistakes instead of absorbing them silently like the old
    `suppress(Exception)` did."""
    from dqliteclient.pool import _POOL_CLEANUP_EXCEPTIONS

    assert AttributeError not in _POOL_CLEANUP_EXCEPTIONS
    assert TypeError not in _POOL_CLEANUP_EXCEPTIONS
