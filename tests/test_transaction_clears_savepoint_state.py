"""``transaction()`` exit must clear the savepoint state alongside
``_in_transaction`` / ``_tx_owner`` — the all-four-clear discipline.

Two failure modes were under-covered:

- The benign no-transaction-active rollback branch (returns silently
  without invalidating) only logged; it did not scrub
  ``_savepoint_stack`` or ``_savepoint_implicit_begin``. The pool-reset
  predicate then saw a stale stack and re-issued a (also-benign)
  ROLLBACK on connection return — wasted round-trip.
- The exit ``finally`` clause cleared only the two transaction fields.
  In practice the success path's ``COMMIT`` and the failure-path
  branches above all clear the savepoint pair before reaching here, but
  the invariant should be local to ``transaction()``'s exit so a future
  refactor that splits ``COMMIT`` from state-update cannot silently
  regress.

These tests pin the all-four-clear discipline at both sites.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


@pytest.mark.asyncio
async def test_benign_no_tx_rollback_clears_savepoint_state() -> None:
    """Benign 'no transaction is active' rollback must clear the
    savepoint pair — server confirms the tx is gone, so any local
    SAVEPOINT tracker is stale by definition."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    no_tx_error = OperationalError(1, "cannot rollback - no transaction is active")

    async def fake_execute(sql: str, params: object = None) -> tuple[int, int]:
        if sql == "BEGIN":
            # Simulate a SAVEPOINT having been pushed onto the stack
            # mid-body before the rollback fires.
            conn._savepoint_stack.append("sp")
            conn._savepoint_implicit_begin = True
            return (0, 0)
        if sql == "ROLLBACK":
            raise no_tx_error
        return (0, 0)

    conn.execute = fake_execute  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise RuntimeError("body raised")

    # Connection preserved (benign branch) AND savepoint state scrubbed.
    assert conn._invalidation_cause is None
    assert conn._protocol is not None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    # Two-field clears were already in place — pin them too.
    assert conn._in_transaction is False
    assert conn._tx_owner is None


@pytest.mark.asyncio
async def test_finally_clears_savepoint_state_on_body_exception_with_invalidation() -> None:
    """When ROLLBACK fails with a non-benign error, the finally clause
    runs after _invalidate has already cleared the pair. The defence-
    in-depth clears in finally must remain idempotent — pin that the
    state stays cleared (no AttributeError, no flap)."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_execute(sql: str, params: object = None) -> tuple[int, int]:
        if sql == "BEGIN":
            conn._savepoint_stack.append("sp1")
            conn._savepoint_stack.append("sp2")
            conn._savepoint_implicit_begin = True
            return (0, 0)
        if sql == "ROLLBACK":
            raise OperationalError(1, "some other failure mode")
        return (0, 0)

    conn.execute = fake_execute  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise RuntimeError("body raised")

    # _invalidate cleared the pair; finally re-cleared idempotently.
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
async def test_finally_clears_savepoint_state_on_success_path() -> None:
    """Pin the defence-in-depth clears on the success path: COMMIT's
    ``_update_tx_flags_from_sql`` already clears the pair, but the
    finally must keep enforcing the invariant. Future refactors that
    split COMMIT from state-update cannot silently regress."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_execute(sql: str, params: object = None) -> tuple[int, int]:
        if sql == "BEGIN":
            conn._savepoint_stack.append("sp")
            conn._savepoint_implicit_begin = True
            return (0, 0)
        if sql == "COMMIT":
            return (0, 0)
        return (0, 0)

    conn.execute = fake_execute  # type: ignore[method-assign]

    async with conn.transaction():
        pass

    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    assert conn._in_transaction is False
    assert conn._tx_owner is None
