"""Pin: ``RELEASE`` and ``ROLLBACK TO`` of a parser-rejected
(quoted, backticked, bracketed, unicode, leading-digit) savepoint
name handle the local stack conservatively so a later operation
on a tracked name doesn't index into a ghost frame.

When the SAVEPOINT name cannot be canonicalised (`_parse_savepoint_name`
returns None), the local tracker abstains from pushing onto
``_savepoint_stack`` and instead sets sticky
``_has_untracked_savepoint=True``. The pool-reset predicate uses the
flag to keep the slot from being reused with stale state — that part
worked.

The gap: when a subsequent ``RELEASE <quoted>`` or
``ROLLBACK TO <quoted>`` arrived, the tracker fell through silently
without updating the local stack. The server, however, executed the
operation. Subsequent operations on tracked names then hit "no such
savepoint" or worse — silent corruption — because the local stack
believed frames existed that the server had popped.

The fix:

* ``RELEASE <quoted>`` — clear the local stack entirely (any tracked
  frame may have been popped by the server's RELEASE; we cannot know
  which without a name to compare). Lock the sticky flag.
* ``ROLLBACK TO <quoted>`` — leave the local stack alone (ROLLBACK
  TO does NOT pop the named savepoint per SQLite spec; only frames
  ABOVE it). Lock the sticky flag.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestReleaseUntrackedClearsLocalStack:
    def test_mixed_quoted_then_tracked_then_release_quoted(self, conn: DqliteConnection) -> None:
        """Concrete scenario from the issue file:

        - ``SAVEPOINT "Outer"`` → server: ["Outer"]; client: stack=[],
          untracked=True.
        - ``SAVEPOINT inner`` → server: ["Outer","inner"]; client:
          stack=["inner"].
        - ``RELEASE "Outer"`` → server: []; client must clear its
          stack so the next operation doesn't see a ghost ["inner"].
        """
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        assert conn._savepoint_stack == []
        assert conn._has_untracked_savepoint is True

        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        assert conn._savepoint_stack == ["inner"]

        conn._update_tx_flags_from_sql('RELEASE "Outer"')
        assert conn._savepoint_stack == [], (
            "RELEASE of an untracked outer SP must conservatively "
            "clear the local stack — without this, the next operation "
            "on the still-tracked 'inner' would be a ghost reference"
        )
        assert conn._has_untracked_savepoint is True, (
            "sticky flag must remain set so pool reset still fires"
        )

    def test_release_untracked_with_empty_stack_is_safe(self, conn: DqliteConnection) -> None:
        """Negative pin: when the stack was already empty (the only
        SP issued was untracked), the conservative clear is a
        well-defined no-op on the stack and the flag stays set."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql('RELEASE "Quoted"')
        assert conn._savepoint_stack == []
        assert conn._has_untracked_savepoint is True

    def test_release_with_savepoint_keyword_quoted_clears_stack(
        self, conn: DqliteConnection
    ) -> None:
        """The grammar accepts ``RELEASE SAVEPOINT name`` as a
        synonym for ``RELEASE name``; the keyword variant must take
        the same conservative path when the name is unparsable."""
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('RELEASE SAVEPOINT "Outer"')
        assert conn._savepoint_stack == []
        assert conn._has_untracked_savepoint is True


class TestRollbackToUntrackedLeavesLocalStack:
    def test_rollback_to_quoted_does_not_clear_local_stack(self, conn: DqliteConnection) -> None:
        """``ROLLBACK TO <name>`` does NOT pop the named savepoint;
        it only unwinds frames ABOVE it. We can't know which (if any)
        of our tracked frames sit above an unparsable target, so the
        conservative answer is to leave the local stack as-is and
        rely on the sticky flag for pool-reset correctness."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TO "Outer"')

        # Server: ["Outer"] (inner unwound). Client: ["inner"]
        # (left alone — over-correction would drop frames that may
        # actually still exist on the server in the case where every
        # tracked frame sits BELOW the target).
        assert conn._savepoint_stack == ["inner"]
        assert conn._has_untracked_savepoint is True

    def test_rollback_to_quoted_with_savepoint_keyword(self, conn: DqliteConnection) -> None:
        """``ROLLBACK TO SAVEPOINT name`` synonym."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TO SAVEPOINT "Outer"')
        assert conn._savepoint_stack == ["inner"]
        assert conn._has_untracked_savepoint is True

    def test_rollback_transaction_to_quoted(self, conn: DqliteConnection) -> None:
        """``ROLLBACK TRANSACTION TO name`` — TRANSACTION keyword
        is optional in BOTH the full-rollback and the rollback-to-
        savepoint forms; the parser strips it before testing for TO."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TRANSACTION TO "Outer"')
        assert conn._savepoint_stack == ["inner"]
        assert conn._has_untracked_savepoint is True


class TestStickyFlagSurvivesUntrackedReleaseRollback:
    def test_pool_reset_still_fires_after_untracked_release(self, conn: DqliteConnection) -> None:
        """End-to-end pool-reset invariant: after an untracked
        RELEASE the slot must NOT be reused without a ROLLBACK."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('RELEASE "Quoted"')

        # Pool reset predicate: any of the four flags = needs reset.
        needs_reset = (
            conn._in_transaction
            or bool(conn._savepoint_stack)
            or conn._savepoint_implicit_begin
            or conn._has_untracked_savepoint
        )
        assert needs_reset is True

    def test_pool_reset_still_fires_after_untracked_rollback_to(
        self, conn: DqliteConnection
    ) -> None:
        """Same invariant on the ROLLBACK TO arm."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TO "Quoted"')

        needs_reset = (
            conn._in_transaction
            or bool(conn._savepoint_stack)
            or conn._savepoint_implicit_begin
            or conn._has_untracked_savepoint
        )
        assert needs_reset is True

    def test_explicit_rollback_clears_sticky_flag(self, conn: DqliteConnection) -> None:
        """The sticky flag is cleared by an explicit ROLLBACK of the
        outer transaction — same contract as before this fix."""
        conn._in_transaction = True
        conn._tx_owner = None
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql('RELEASE "Quoted"')
        assert conn._has_untracked_savepoint is True

        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._has_untracked_savepoint is False
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
