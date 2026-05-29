"""Multi-statement EXEC failing mid-batch sets ``_has_untracked_savepoint`` so pool
reset's ROLLBACK fires: the success-only ``_update_tx_flags_from_sql`` is skipped on
raise, so on failure we set the flag whenever the SQL has ``;`` and any tx-control verb."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


async def _raise_constraint(_fn: Any) -> Any:
    raise OperationalError("constraint failed", 19)


def _fake_connected_conn() -> DqliteConnection:
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]
    return conn


@pytest.mark.asyncio
async def test_multi_statement_exec_partial_failure_sets_untracked_flag() -> None:
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("BEGIN; SAVEPOINT a; INSERT INTO t VALUES (?)", (1,))
    assert conn._has_untracked_savepoint is True


@pytest.mark.asyncio
async def test_multi_statement_no_tx_verbs_does_not_set_untracked_flag() -> None:
    """A benign multi-INSERT batch (no tx-control verb) does NOT set the flag."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute(
            "INSERT INTO t VALUES (?); INSERT INTO t VALUES (?); INSERT INTO t VALUES (?)",
            (1,),
        )
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_single_statement_failure_does_not_set_flag() -> None:
    """A single-statement EXEC failure (no ``;``) skips the flag path."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("INSERT INTO t VALUES (?)", (1,))
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_multi_statement_with_savepoint_only_sets_flag() -> None:
    """SAVEPOINT alone (no other tx verb) still sets the flag; it can autobegin a tx."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("SAVEPOINT sp; INSERT INTO t VALUES (?)", (1,))
    assert conn._has_untracked_savepoint is True
