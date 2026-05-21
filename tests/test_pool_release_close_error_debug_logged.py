"""Pin: the seven release/drain close sites that previously used
``contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS)`` now route through
``ConnectionPool._close_best_effort``, which logs the absorbed
exception at DEBUG with ``exc_info=True`` rather than silently
dropping it.

Pre-fix the sites suppressed silently — an OSError /
DqliteConnectionError / OperationalError from ``conn.close()`` during
release/drain dissolved with no trace, asymmetric with
``_initialize_close_unqueued`` (which logged via ``exc_info=True``).
An operator investigating "the pool kept hitting close_timeout under
load" had nothing to find at any verbosity.

This test drives the closed-pool release branch (representative of the
seven sites) and asserts a DEBUG record naming the site lands with
``exc_info`` populated. Pairs with
``test_pool_initialize_unqueued_survivor_close_log.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
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


def _make_pool_with_fake_cluster(fake_conn: _FakeConn) -> ConnectionPool:
    async def _connect(**kwargs: Any) -> _FakeConn:
        return fake_conn

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _connect
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_release_closed_branch_logs_close_error_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The closed-pool release branch routes through
    ``_close_best_effort``: a ``DqliteConnectionError`` from
    ``conn.close()`` is absorbed but emits a DEBUG record naming the
    site (``"pool.release-closed: close error"``) with ``exc_info``
    populated."""
    fake = _FakeConn(close_side_effect=DqliteConnectionError("transport broken"))
    pool = _make_pool_with_fake_cluster(fake)

    cm = pool.acquire()
    conn = await cm.__aenter__()
    assert conn is fake
    pool._closed = True

    with caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"):
        # Must NOT raise — the helper absorbs _POOL_CLEANUP_EXCEPTIONS.
        await cm.__aexit__(None, None, None)

    assert fake.close_calls >= 1

    # A DEBUG record named for the site, with exc_info populated.
    debug_records = [
        r for r in caplog.records if r.levelno == logging.DEBUG and "close error" in r.message
    ]
    assert debug_records, (
        "expected a DEBUG record naming the release site; "
        f"saw records={[(r.levelname, r.message) for r in caplog.records]}"
    )
    matched = [r for r in debug_records if "release-closed" in r.message]
    assert matched, (
        f"expected the release-closed site name in the DEBUG record; "
        f"got messages={[r.message for r in debug_records]}"
    )
    rec = matched[0]
    assert rec.exc_info is not None
    assert isinstance(rec.exc_info[1], DqliteConnectionError)


def test_close_best_effort_helper_exists_with_logging_shape() -> None:
    """Static pin: ``ConnectionPool`` exposes ``_close_best_effort``, the
    DRY helper that replaced the seven ``contextlib.suppress(
    *_POOL_CLEANUP_EXCEPTIONS)`` sites. Detects accidental removal of
    the helper or rename that would silently drop the logging on the
    release path again."""
    assert hasattr(ConnectionPool, "_close_best_effort")
