"""A deferred-FK error (SQLITE_CONSTRAINT 19) on plain COMMIT/END or RELEASE of the
outermost savepoint clears tracker state: SQLite treats it as COMMIT and tears down
the whole tx, yet code 19 is NOT in the general auto-rollback set, so verb-condition it."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


async def _raise_constraint(_fn: Any) -> Any:
    raise OperationalError("FOREIGN KEY constraint failed", 19)


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
    """``RELEASE SAVEPOINT outer`` form (optional SAVEPOINT keyword)."""
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
    """END is the SQLite synonym for COMMIT."""
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("END TRANSACTION")
    assert conn._in_transaction is False


@pytest.mark.asyncio
async def test_multistatement_deferred_fk_release_outermost_clears_tracker() -> None:
    """Multi-statement EXEC ending in RELEASE-outermost (code 19) triggers the clear;
    the classifier reads the trailing piece (gateway stops on first failure)."""
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("BEGIN; INSERT INTO t VALUES (1); RELEASE outer")
    assert conn._in_transaction is False
    assert conn._savepoint_stack == []
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_multistatement_auto_rollback_does_not_re_set_untracked_flag() -> None:
    """After _run_protocol clears the tracker via the auto-rollback branch, the
    multi-statement conservative flag must NOT re-set _has_untracked_savepoint."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]
    conn._in_transaction = True
    conn._savepoint_stack = ["outer"]
    conn._savepoint_implicit_begin = True

    async def _raise_interrupt(*args: object, **kwargs: object) -> None:
        # SQLITE_INTERRUPT (9) auto-rollback: mimic _run_protocol clearing the tracker.
        conn._in_transaction = False
        conn._savepoint_stack.clear()
        conn._savepoint_implicit_begin = False
        conn._has_untracked_savepoint = False
        raise OperationalError("interrupted", 9)

    with (
        patch.object(conn, "_run_protocol", new=_raise_interrupt),
        pytest.raises(OperationalError),
    ):
        await conn.execute("BEGIN; INSERT INTO t VALUES (1)")
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_commit_with_constraint_failure_outside_tx_does_not_clear() -> None:
    """Code 19 on COMMIT outside a tx must NOT clear; the classifier requires
    ``_in_transaction`` to be True."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]
    conn._in_transaction = False
    assert conn._sql_is_outermost_release_or_commit("COMMIT") is False
    assert conn._sql_is_outermost_release_or_commit("END TRANSACTION") is False
    conn._in_transaction = True
    assert conn._sql_is_outermost_release_or_commit("COMMIT") is True
    assert conn._sql_is_outermost_release_or_commit("END TRANSACTION") is True


@pytest.mark.asyncio
async def test_release_inner_savepoint_with_constraint_failure_keeps_tracker() -> None:
    """RELEASE of an inner savepoint with constraint failure preserves tracker state."""
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
    assert conn._in_transaction is True
    assert conn._savepoint_stack == ["outer", "inner"]


@pytest.mark.asyncio
async def test_constraint_failure_on_plain_insert_keeps_tracker() -> None:
    """A plain INSERT with constraint failure does NOT auto-rollback."""
    conn = _conn_with_outermost_savepoint("outer")
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("INSERT INTO t VALUES (?)", (1,))
    assert conn._in_transaction is True
    assert conn._savepoint_stack == ["outer"]
