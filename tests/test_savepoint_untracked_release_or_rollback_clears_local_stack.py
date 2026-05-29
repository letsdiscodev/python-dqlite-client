"""``RELEASE``/``ROLLBACK TO`` of an unparsable (quoted/unicode/...) savepoint name
handle the local stack conservatively so later tracked-name ops don't hit ghost frames:
RELEASE clears the whole stack, ROLLBACK TO leaves it alone; both lock the sticky flag."""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestReleaseUntrackedClearsLocalStack:
    def test_mixed_quoted_then_tracked_then_release_quoted(self, conn: DqliteConnection) -> None:
        """Untracked outer + tracked inner, then RELEASE outer: client must clear
        its stack so the next op doesn't see a ghost ["inner"]."""
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
        """When the stack was already empty, the conservative clear is a no-op."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql('RELEASE "Quoted"')
        assert conn._savepoint_stack == []
        assert conn._has_untracked_savepoint is True

    def test_release_with_savepoint_keyword_quoted_clears_stack(
        self, conn: DqliteConnection
    ) -> None:
        """``RELEASE SAVEPOINT name`` synonym takes the same conservative path."""
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('RELEASE SAVEPOINT "Outer"')
        assert conn._savepoint_stack == []
        assert conn._has_untracked_savepoint is True


class TestRollbackToUntrackedLeavesLocalStack:
    def test_rollback_to_quoted_does_not_clear_local_stack(self, conn: DqliteConnection) -> None:
        """ROLLBACK TO unwinds only frames ABOVE the target; with an unparsable
        name we leave the local stack as-is and rely on the sticky flag."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TO "Outer"')

        # Left alone: over-correction would drop frames that may still exist
        # if every tracked frame sits BELOW the target.
        assert conn._savepoint_stack == ["inner"]
        assert conn._has_untracked_savepoint is True

    def test_rollback_to_quoted_with_savepoint_keyword(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TO SAVEPOINT "Outer"')
        assert conn._savepoint_stack == ["inner"]
        assert conn._has_untracked_savepoint is True

    def test_rollback_transaction_to_quoted(self, conn: DqliteConnection) -> None:
        """``ROLLBACK TRANSACTION TO name`` — the optional TRANSACTION keyword
        is stripped before testing for TO."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Outer"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('ROLLBACK TRANSACTION TO "Outer"')
        assert conn._savepoint_stack == ["inner"]
        assert conn._has_untracked_savepoint is True


class TestStickyFlagSurvivesUntrackedReleaseRollback:
    def test_pool_reset_still_fires_after_untracked_release(self, conn: DqliteConnection) -> None:
        """After an untracked RELEASE the slot must NOT be reused without a ROLLBACK."""
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql('RELEASE "Quoted"')

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
        """An explicit ROLLBACK of the outer transaction clears the sticky flag."""
        conn._in_transaction = True
        conn._tx_owner = None
        conn._update_tx_flags_from_sql('SAVEPOINT "Quoted"')
        conn._update_tx_flags_from_sql('RELEASE "Quoted"')
        assert conn._has_untracked_savepoint is True

        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._has_untracked_savepoint is False
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
