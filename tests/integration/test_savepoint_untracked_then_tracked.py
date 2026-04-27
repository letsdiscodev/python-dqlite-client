"""Integration test for the SAVEPOINT untracked-then-tracked sequence
against the real dqlite cluster.

The client-side parser deliberately rejects quoted / backtick /
square-bracket / unicode SAVEPOINT names — the case-sensitivity
trade-off in ``_parse_savepoint_name`` is documented to ensure
identifier handling stays consistent across the codebase. The
upstream server, by contrast, has no such restriction: it forwards
the SQL to the embedded SQLite engine, which auto-begins a
transaction on ANY well-formed ``SAVEPOINT`` statement.

The mismatch is the load-bearing case for the post-fix invariants:

1. ``_has_untracked_savepoint`` flips True (parser couldn't
   represent the name).
2. ``_savepoint_implicit_begin`` STAYS False (the auto-begin was
   driven by the outer untracked SAVEPOINT, not by the inner
   tracked one — there is no implicit begin to record).
3. ``in_transaction`` returns True via the property's OR over
   ``_in_transaction or _has_untracked_savepoint``.
4. After the cleanup ROLLBACK clears the autobegun tx, both
   flags drop and a follow-up ``transaction()`` ctxmgr can enter
   without raising.

The unit-level tests against mocks pin (1)-(3) but cannot verify
that the upstream server actually auto-begins on a quoted
SAVEPOINT — only the live cluster confirms that. This integration
test closes the loop end-to-end.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection


@pytest.mark.integration
@pytest.mark.asyncio
async def test_quoted_savepoint_then_tracked_savepoint_then_release(
    cluster_address: str,
) -> None:
    """Drive the full ``SAVEPOINT "Foo"`` → ``SAVEPOINT inner`` →
    ``RELEASE inner`` sequence and pin the post-fix client-side
    invariants at every step."""
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()

        # Baseline: clean connection, no transaction.
        assert conn._in_transaction is False
        assert conn._has_untracked_savepoint is False
        assert conn._savepoint_implicit_begin is False
        assert conn.in_transaction is False

        # Quoted SAVEPOINT — the client parser refuses to extract
        # the name (case-sensitive identifier rule), so the tracker
        # marks the connection as having an untracked savepoint and
        # leaves the explicit-tx flag alone. The server, however,
        # auto-begins a transaction on the wire.
        await conn.execute('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True, (
            "quoted SAVEPOINT name must trip the untracked flag — the parser cannot represent it"
        )
        assert conn._in_transaction is False, (
            "the parser-rejected name leaves the explicit-tx flag "
            "untouched (no BEGIN was issued, only the SAVEPOINT)"
        )
        assert conn.in_transaction is True, (
            "in_transaction property must OR in the untracked flag — "
            "the server-side autobegun tx is real and the property "
            "is the load-bearing read"
        )

        # Tracked inner SAVEPOINT. POST-FIX INVARIANT:
        # ``_savepoint_implicit_begin`` stays False. Pre-fix it
        # would be set on the inner SAVEPOINT because the tracker
        # didn't know the outer untracked one had already
        # auto-begun the transaction. Setting it would lead to an
        # incorrect ROLLBACK target on cleanup.
        await conn.execute("SAVEPOINT inner")
        assert conn._savepoint_implicit_begin is False, (
            "inner tracked SAVEPOINT must NOT record an implicit "
            "begin — the outer untracked SAVEPOINT already auto-began"
        )
        assert conn._has_untracked_savepoint is True, (
            "untracked flag survives an inner tracked SAVEPOINT"
        )

        # Release the inner. Server-side: outer "Foo" still alive,
        # so the autobegun tx is still alive too.
        await conn.execute("RELEASE inner")
        assert conn.in_transaction is True, (
            "outer SAVEPOINT still holds the autobegun tx after releasing the inner one"
        )
        assert conn._has_untracked_savepoint is True

        # Cleanup ROLLBACK clears the autobegun tx and the
        # untracked flag.
        await conn.execute("ROLLBACK")
        assert conn._in_transaction is False
        assert conn._has_untracked_savepoint is False
        assert conn._savepoint_implicit_begin is False
        assert conn.in_transaction is False
    finally:
        await conn.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transaction_ctxmgr_enters_cleanly_after_untracked_cleanup(
    cluster_address: str,
) -> None:
    """Pin the follow-up: after the cleanup ROLLBACK clears
    ``_has_untracked_savepoint``, a fresh ``transaction()`` ctxmgr
    can enter without raising the "SAVEPOINT outside an explicit
    BEGIN" InterfaceError. Catches a regression that forgets to
    clear the flag on ROLLBACK and leaves the slot poisoned for
    every subsequent caller."""
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()
        await conn.execute('SAVEPOINT "Bar"')
        assert conn._has_untracked_savepoint is True

        await conn.execute("ROLLBACK")
        assert conn._has_untracked_savepoint is False, (
            "ROLLBACK must clear ``_has_untracked_savepoint``; "
            "without this clear, the next transaction() ctxmgr "
            "would see the stale flag and refuse to enter"
        )

        # Fresh transaction() ctxmgr enters cleanly — no
        # InterfaceError because the flag was cleared above.
        async with conn.transaction():
            assert conn._in_transaction is True
        assert conn._in_transaction is False
    finally:
        await conn.close()
