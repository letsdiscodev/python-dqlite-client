"""Pin SQLite's LIFO duplicate-name semantics across the SAVEPOINT /
RELEASE / ROLLBACK TO operations.

Per https://www.sqlite.org/lang_savepoint.html:

    "The name of a savepoint need not be unique. If multiple
    savepoints have the same name, then SQLite uses the most recently
    created savepoint with the matching name."

The tracker implements this via reverse-search; this file exercises
the contract across multi-step operation sequences so a future
refactor cannot silently regress duplicate-name handling.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    c = DqliteConnection("localhost:9001")
    c._db_id = 1
    c._protocol = object()  # type: ignore[assignment]
    return c


def _drive(conn: DqliteConnection, ops: list[str]) -> None:
    for op in ops:
        conn._update_tx_flags_from_sql(op)


def test_duplicate_name_push_records_both_frames(conn: DqliteConnection) -> None:
    _drive(conn, ["SAVEPOINT a", "SAVEPOINT b", "SAVEPOINT a"])
    assert conn._savepoint_stack == ["a", "b", "a"]


def test_release_of_duplicate_name_pops_most_recent(conn: DqliteConnection) -> None:
    _drive(conn, ["SAVEPOINT a", "SAVEPOINT b", "SAVEPOINT a", "RELEASE a"])
    assert conn._savepoint_stack == ["a", "b"]


def test_release_of_outer_duplicate_pops_above_too(conn: DqliteConnection) -> None:
    _drive(conn, ["SAVEPOINT a", "SAVEPOINT b", "SAVEPOINT a", "RELEASE b"])
    # RELEASE pops the named SP and every frame above it. The most-
    # recent ``a`` sits above ``b``; both go.
    assert conn._savepoint_stack == ["a"]


def test_rollback_to_duplicate_targets_innermost(conn: DqliteConnection) -> None:
    _drive(conn, ["SAVEPOINT a", "SAVEPOINT b", "SAVEPOINT a", "ROLLBACK TO a"])
    # ROLLBACK TO leaves the named SP active; nothing above the most-
    # recent ``a`` to pop.
    assert conn._savepoint_stack == ["a", "b", "a"]


def test_rollback_to_outer_duplicate_pops_above(conn: DqliteConnection) -> None:
    _drive(conn, ["SAVEPOINT a", "SAVEPOINT b", "SAVEPOINT a", "ROLLBACK TO b"])
    # Pops above ``b``; ``b`` and ``a`` (outer) survive, the inner
    # ``a`` is gone.
    assert conn._savepoint_stack == ["a", "b"]


def test_three_duplicates_release_innermost(conn: DqliteConnection) -> None:
    _drive(
        conn,
        ["SAVEPOINT a", "SAVEPOINT a", "SAVEPOINT a", "RELEASE a"],
    )
    assert conn._savepoint_stack == ["a", "a"]


def test_three_duplicates_rollback_to_outermost(conn: DqliteConnection) -> None:
    # ROLLBACK TO targets the most-recently-created with the matching
    # name (LIFO), not the outermost. Frames above are popped.
    _drive(
        conn,
        [
            "SAVEPOINT a",
            "SAVEPOINT b",
            "SAVEPOINT a",
            "SAVEPOINT c",
            "ROLLBACK TO a",
        ],
    )
    # ``c`` popped (above the most-recent ``a``); inner ``a``, ``b``,
    # outer ``a`` all stay.
    assert conn._savepoint_stack == ["a", "b", "a"]
