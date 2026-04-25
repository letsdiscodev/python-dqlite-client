"""DqliteConnection's _in_transaction flag must track raw BEGIN /
COMMIT / ROLLBACK statements, not only the transaction() context
manager. The dbapi's in_transaction property delegates to this flag
and the documented contract is parity with stdlib
sqlite3.Connection.in_transaction.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection


@pytest.mark.integration
@pytest.mark.asyncio
async def test_raw_begin_sets_in_transaction(cluster_address: str) -> None:
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()
        assert conn._in_transaction is False
        await conn.execute("BEGIN")
        assert conn._in_transaction is True
        await conn.execute("COMMIT")
        assert conn._in_transaction is False
    finally:
        await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_raw_rollback_clears_in_transaction(cluster_address: str) -> None:
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()
        await conn.execute("BEGIN")
        assert conn._in_transaction is True
        await conn.execute("ROLLBACK")
        assert conn._in_transaction is False
    finally:
        await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_savepoint_does_not_toggle_in_transaction(cluster_address: str) -> None:
    """SAVEPOINT / RELEASE / ROLLBACK TO leave the outer tx boundary
    unchanged; in_transaction stays True throughout."""
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()
        await conn.execute("DROP TABLE IF EXISTS test_sp_raw")
        await conn.execute("CREATE TABLE test_sp_raw (n INTEGER PRIMARY KEY)")
        await conn.execute("BEGIN")
        await conn.execute("SAVEPOINT sp1")
        assert conn._in_transaction is True
        await conn.execute("RELEASE SAVEPOINT sp1")
        assert conn._in_transaction is True
        await conn.execute("COMMIT")
        assert conn._in_transaction is False
    finally:
        await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_begin_qualifiers_recognised(cluster_address: str) -> None:
    """BEGIN DEFERRED / IMMEDIATE / EXCLUSIVE all qualify as starting
    a transaction."""
    for qualifier in ("BEGIN DEFERRED", "BEGIN IMMEDIATE", "BEGIN EXCLUSIVE"):
        conn = DqliteConnection(cluster_address)
        try:
            await conn.connect()
            await conn.execute(qualifier)
            assert conn._in_transaction is True, f"{qualifier} did not set the flag"
            await conn.execute("ROLLBACK")
            assert conn._in_transaction is False
        finally:
            await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_recognised_as_commit(cluster_address: str) -> None:
    """SQLite's ``END`` is a synonym for ``COMMIT``; the flag tracker
    must recognise it."""
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()
        await conn.execute("BEGIN")
        await conn.execute("END")
        assert conn._in_transaction is False
    finally:
        await conn.close()
