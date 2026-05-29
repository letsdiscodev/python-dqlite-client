"""_reset_connection logs leader-flip ROLLBACK failures
(LEADER_ERROR_CODES, routine churn) at DEBUG and genuine faults at
WARNING with traceback.
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
    """Fake conn routed through the ROLLBACK arm (in-tx, protocol looks healthy)."""
    conn = MagicMock()
    conn._address = "localhost:9001"
    conn._in_transaction = True
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
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
    """A LEADER_ERROR_CODES ROLLBACK failure logs DEBUG, not WARNING."""
    pool = _make_pool()
    leader_code = next(iter(LEADER_ERROR_CODES))
    err = OperationalError("not leader", leader_code)
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
    """A non-LEADER_ERROR_CODES ROLLBACK failure logs WARNING with traceback."""
    pool = _make_pool()
    non_leader_code = 1
    assert non_leader_code not in LEADER_ERROR_CODES
    err = OperationalError("disk I/O error", non_leader_code)
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
    assert warning_records[0].exc_info is not None, (
        "non-leader ROLLBACK failure WARNING must include exc_info=True"
    )
