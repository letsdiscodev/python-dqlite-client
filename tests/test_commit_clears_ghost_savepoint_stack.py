"""COMMIT/END clears the savepoint stack even when ``_in_transaction`` is False.

An untracked outer SAVEPOINT defers autobegin, leaving stack non-empty with
_in_transaction=False; COMMIT/END must still clear it (mirroring ROLLBACK) or a
ghost frame leaks into RELEASE and the pool-return safety predicate.
"""

from __future__ import annotations

from dqliteclient import DqliteConnection


def _make_connection() -> DqliteConnection:
    """No ``connect()`` so only the pure state machine is exercised."""
    return DqliteConnection("127.0.0.1:9001")


def test_commit_clears_savepoint_stack_after_untracked_deferred_autobegin() -> None:
    conn = _make_connection()
    conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
    assert conn._has_untracked_savepoint is True
    assert conn._savepoint_stack == []
    assert conn._in_transaction is False

    conn._update_tx_flags_from_sql("SAVEPOINT inner")
    # Autobegin deferred (untracked); stack tracks inner despite not-in-tx.
    assert conn._savepoint_stack == ["inner"]
    assert conn._in_transaction is False

    conn._update_tx_flags_from_sql("COMMIT")
    assert conn._savepoint_stack == [], (
        "COMMIT must clear ghost savepoint stack frames even when "
        "_in_transaction is False (mirrors ROLLBACK's defensive clear)"
    )
    assert conn._has_untracked_savepoint is False
    assert conn._in_transaction is False
    assert conn._savepoint_implicit_begin is False


def test_end_clears_savepoint_stack_after_untracked_deferred_autobegin() -> None:
    """END aliases COMMIT in SQLite; same defensive clear applies."""
    conn = _make_connection()
    conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
    conn._update_tx_flags_from_sql("SAVEPOINT inner")
    assert conn._savepoint_stack == ["inner"]

    conn._update_tx_flags_from_sql("END")
    assert conn._savepoint_stack == []


def test_commit_in_active_tx_still_clears_state() -> None:
    """The existing in-tx COMMIT path is unchanged."""
    conn = _make_connection()
    conn._update_tx_flags_from_sql("BEGIN")
    conn._update_tx_flags_from_sql("SAVEPOINT a")
    assert conn._in_transaction is True
    assert conn._savepoint_stack == ["a"]

    conn._update_tx_flags_from_sql("COMMIT")
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    assert conn._has_untracked_savepoint is False
