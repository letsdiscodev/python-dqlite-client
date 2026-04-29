"""Pin: ``ConnectionPool._reset_connection`` distinguishes leader-flip
ROLLBACK failures (DEBUG) from genuine server faults (WARNING).

A code in ``LEADER_ERROR_CODES`` represents routine cluster churn —
the new leader has no record of our transaction, so the ROLLBACK
fails predictably. Operators should not get a WARNING-level log line
per leader flip on a busy pool. Codes outside that set indicate an
actual fault and DO warrant the WARNING + traceback so the operator
can investigate.

Without this test, a future refactor that flattened the two arms to
the same level (or swapped them) would silently degrade the
operational signal.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import OperationalError
from dqliteclient.pool import ConnectionPool
from dqlitewire import LEADER_ERROR_CODES


def _make_pool() -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )


def _conn_in_tx(execute_side_effect: BaseException) -> Any:
    """Build a fake conn that ``_reset_connection`` will route through
    the ROLLBACK arm. ``_in_transaction=True`` selects the arm; the
    socket-liveness check is bypassed by leaving the protocol stub
    looking healthy."""
    conn = MagicMock()
    conn._address = "localhost:9001"
    conn._in_transaction = True
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    # Look-alive for the cheap pre-write liveness check.
    conn._protocol = MagicMock()
    conn._protocol._writer = MagicMock()
    conn._protocol._writer.transport = MagicMock()
    conn._protocol._writer.transport.is_closing = lambda: False
    conn._protocol._reader = MagicMock()
    conn._protocol._reader.at_eof = lambda: False
    conn._protocol.is_wire_coherent = True
    conn.execute = AsyncMock(side_effect=execute_side_effect)
    return conn


@pytest.mark.asyncio
async def test_reset_leader_class_rollback_failure_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ROLLBACK failure with a ``LEADER_ERROR_CODES`` code logs at
    DEBUG, not WARNING — routine cluster churn shouldn't trigger
    operator alerting."""
    pool = _make_pool()
    leader_code = next(iter(LEADER_ERROR_CODES))
    err = OperationalError(leader_code, "not leader")
    conn = _conn_in_tx(err)

    with caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"):
        result = await pool._reset_connection(conn)

    assert result is False, "leader-class ROLLBACK failure must drop the conn"
    debug_records = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "leader-class ROLLBACK failure" in r.message
    ]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert debug_records, (
        f"expected a DEBUG record about leader-class ROLLBACK failure; "
        f"got {[(r.levelno, r.message) for r in caplog.records]!r}"
    )
    assert not warning_records, (
        "leader-class ROLLBACK failure must NOT produce a WARNING — "
        "routine cluster churn shouldn't trigger operator alerting; "
        f"got {[(r.levelno, r.message) for r in warning_records]!r}"
    )


@pytest.mark.asyncio
async def test_reset_non_leader_rollback_failure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ROLLBACK failure with a code OUTSIDE ``LEADER_ERROR_CODES``
    is a genuine server fault and logs at WARNING with traceback so
    the operator can investigate."""
    pool = _make_pool()
    # Pick a code that is NOT in LEADER_ERROR_CODES.
    non_leader_code = 1
    assert non_leader_code not in LEADER_ERROR_CODES
    err = OperationalError(non_leader_code, "disk I/O error")
    conn = _conn_in_tx(err)

    with caplog.at_level(logging.WARNING, logger="dqliteclient.pool"):
        result = await pool._reset_connection(conn)

    assert result is False
    warning_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "ROLLBACK failure" in r.message
    ]
    assert warning_records, (
        f"expected a WARNING record about ROLLBACK failure; "
        f"got {[(r.levelno, r.message) for r in caplog.records]!r}"
    )
    # WARNING path must include exc_info for traceback so the operator
    # can investigate the genuine fault.
    assert warning_records[0].exc_info is not None, (
        "non-leader ROLLBACK failure WARNING must include exc_info=True"
    )
