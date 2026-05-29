"""Integration test for the SAVEPOINT untracked-then-tracked sequence.

The client parser rejects quoted SAVEPOINT names but the upstream SQLite
engine still auto-begins a transaction on them; only a live cluster confirms
that the post-fix client-side flag invariants hold against that behaviour.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection


@pytest.mark.integration
@pytest.mark.asyncio
async def test_quoted_savepoint_then_tracked_savepoint_then_release(
    cluster_address: str,
) -> None:
    """Pin client-side flag invariants across SAVEPOINT "Foo" → inner → RELEASE."""
    conn = DqliteConnection(cluster_address)
    try:
        await conn.connect()

        assert conn._in_transaction is False
        assert conn._has_untracked_savepoint is False
        assert conn._savepoint_implicit_begin is False
        assert conn.in_transaction is False

        # Parser rejects the quoted name → untracked flag set, explicit-tx flag
        # untouched; the server still auto-begins a tx on the wire.
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

        # Inner tracked SAVEPOINT must NOT set _savepoint_implicit_begin: the
        # outer untracked one already auto-began (else: wrong ROLLBACK target).
        await conn.execute("SAVEPOINT inner")
        assert conn._savepoint_implicit_begin is False, (
            "inner tracked SAVEPOINT must NOT record an implicit "
            "begin — the outer untracked SAVEPOINT already auto-began"
        )
        assert conn._has_untracked_savepoint is True, (
            "untracked flag survives an inner tracked SAVEPOINT"
        )

        # Releasing inner leaves outer "Foo" and its autobegun tx alive.
        await conn.execute("RELEASE inner")
        assert conn.in_transaction is True, (
            "outer SAVEPOINT still holds the autobegun tx after releasing the inner one"
        )
        assert conn._has_untracked_savepoint is True

        # Cleanup ROLLBACK clears the autobegun tx and the untracked flag.
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
    """After ROLLBACK clears the untracked flag, a fresh transaction() enters
    cleanly; catches a regression that leaves the flag set and poisons the slot."""
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

        async with conn.transaction():
            assert conn._in_transaction is True
        assert conn._in_transaction is False
    finally:
        await conn.close()
