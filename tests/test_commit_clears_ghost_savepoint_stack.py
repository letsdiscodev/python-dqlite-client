"""Pin: COMMIT/END clears the savepoint stack EVEN when
``_in_transaction`` is False.

The earlier autobegin-deferral fix (when an outer untracked
SAVEPOINT had set ``_has_untracked_savepoint=True``) deliberately
allows a state where ``_savepoint_stack`` is non-empty AND
``_in_transaction=False``. The COMMIT/END branch in
``_update_tx_flags_from_sql`` was not updated to match: the
stack-clear is gated on ``_in_transaction``, leaving a ghost frame.

Symmetric ROLLBACK already does the defensive double-clear in both
``if`` and ``else`` arms; COMMIT/END must follow.

Sequence that produces the ghost frame today:
    SAVEPOINT "Foo"   ->  _has_untracked_savepoint=True, stack=[],
                          _in_transaction=False
    SAVEPOINT inner   ->  stack=["inner"], _in_transaction=False
                          (autobegin DEFERRED because of untracked)
    COMMIT             ->  _in_transaction=False; stack="inner"
                           (GHOST), _has_untracked_savepoint=False

User-visible consequences:
1. Subsequent RELEASE inner finds "inner" in the local stack and
   pops it, then the server raises "no such savepoint".
2. The pool-return safety predicate (which ORs ``_in_transaction``
   and ``_has_untracked_savepoint``) sees neither flag and skips
   the defensive ROLLBACK; the next acquirer reuses stale state.
"""

from __future__ import annotations

from dqliteclient import DqliteConnection


def _make_connection() -> DqliteConnection:
    """Connection with no ``connect()`` so the pure-state-machine
    code path of ``_update_tx_flags_from_sql`` is exercised."""
    return DqliteConnection("127.0.0.1:9001")


def test_commit_clears_savepoint_stack_after_untracked_deferred_autobegin() -> None:
    conn = _make_connection()
    conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
    assert conn._has_untracked_savepoint is True
    assert conn._savepoint_stack == []
    assert conn._in_transaction is False

    conn._update_tx_flags_from_sql("SAVEPOINT inner")
    # Autobegin is deferred because _has_untracked_savepoint=True;
    # the stack tracks the inner name even though we're "not" in tx.
    assert conn._savepoint_stack == ["inner"]
    assert conn._in_transaction is False

    conn._update_tx_flags_from_sql("COMMIT")
    # The stack must be cleared even though _in_transaction was False.
    # Without this, "inner" remains as a ghost frame.
    assert conn._savepoint_stack == [], (
        "COMMIT must clear ghost savepoint stack frames even when "
        "_in_transaction is False (mirrors ROLLBACK's defensive clear)"
    )
    assert conn._has_untracked_savepoint is False
    assert conn._in_transaction is False
    assert conn._savepoint_implicit_begin is False


def test_end_clears_savepoint_stack_after_untracked_deferred_autobegin() -> None:
    """END is an alias for COMMIT in SQLite. Same defensive clear
    must apply."""
    conn = _make_connection()
    conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
    conn._update_tx_flags_from_sql("SAVEPOINT inner")
    assert conn._savepoint_stack == ["inner"]

    conn._update_tx_flags_from_sql("END")
    assert conn._savepoint_stack == []


def test_commit_in_active_tx_still_clears_state() -> None:
    """Sanity: the existing in-tx COMMIT path is unchanged."""
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
