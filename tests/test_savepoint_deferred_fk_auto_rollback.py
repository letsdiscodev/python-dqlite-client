"""Pin: deferred-FK error on RELEASE of OUTERMOST autobegin SAVEPOINT
(or plain COMMIT) clears tracker state — server has auto-rolled back.

Per https://www.sqlite.org/lang_savepoint.html, an attempt to RELEASE
the outermost savepoint after a deferred-FK error is treated as
COMMIT — the engine tears down the entire transaction. SQLite returns
SQLITE_CONSTRAINT (primary 19), but that code is NOT in the general
auto-rollback set (a CHECK violation on plain INSERT does NOT
auto-rollback). Verb-condition the clear: only fires for plain
COMMIT/END or RELEASE of the outermost-frame savepoint name.

Without this fix, the local view stays at ``in_transaction=True``
while the server has no transaction; pool reset's umbrella ROLLBACK
heals across acquirers (benign no-tx response) but in-task observers
see in_transaction lie and same-task RELEASE retries hit
"no such savepoint."
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


async def _raise_constraint(_fn: Any) -> Any:
    raise OperationalError(19, "FOREIGN KEY constraint failed")


def _conn_with_outermost_savepoint(name: str = "outer") -> DqliteConnection:
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]
    conn._savepoint_stack = [name]
    conn._savepoint_implicit_begin = True
    conn._in_transaction = True
    return conn


@pytest.mark.asyncio
async def test_release_outermost_with_deferred_fk_violation_clears_tracker() -> None:
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError, match="FOREIGN KEY"),
    ):
        await conn.execute("RELEASE outer")
    assert conn._in_transaction is False
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_release_savepoint_outermost_with_deferred_fk_clears_tracker() -> None:
    """``RELEASE SAVEPOINT outer`` form (with optional SAVEPOINT keyword)."""
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("RELEASE SAVEPOINT outer")
    assert conn._in_transaction is False


@pytest.mark.asyncio
async def test_commit_with_deferred_fk_violation_clears_tracker() -> None:
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("COMMIT")
    assert conn._in_transaction is False


@pytest.mark.asyncio
async def test_end_with_deferred_fk_violation_clears_tracker() -> None:
    """END is the SQLite-spec synonym for COMMIT."""
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("END TRANSACTION")
    assert conn._in_transaction is False


@pytest.mark.asyncio
async def test_release_inner_savepoint_with_constraint_failure_keeps_tracker() -> None:
    """Negative pin: RELEASE of an INNER (non-outermost) savepoint
    with constraint failure does NOT auto-rollback the outer tx, so
    tracker state must be preserved."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]
    conn._savepoint_stack = ["outer", "inner"]
    conn._savepoint_implicit_begin = True
    conn._in_transaction = True

    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("RELEASE inner")
    # Inner-frame RELEASE failure does NOT auto-rollback; server keeps
    # the outer tx alive.
    assert conn._in_transaction is True
    assert conn._savepoint_stack == ["outer", "inner"]


@pytest.mark.asyncio
async def test_constraint_failure_on_plain_insert_keeps_tracker() -> None:
    """Negative pin: a plain INSERT with constraint failure does NOT
    auto-rollback. The verb-condition prevents over-clearing."""
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("INSERT INTO t VALUES (?)", (1,))
    # Constraint failure on INSERT does NOT auto-rollback.
    assert conn._in_transaction is True
    assert conn._savepoint_stack == ["outer"]
