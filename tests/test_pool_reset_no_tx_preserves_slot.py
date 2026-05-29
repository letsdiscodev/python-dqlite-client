"""Pool reset preserves the slot when ROLLBACK reports "no tx active"
(healthy conn, stale local flag); any other failure drops the slot.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.connection import _is_no_tx_rollback_error
from dqliteclient.exceptions import OperationalError
from dqliteclient.pool import ConnectionPool


class TestIsNoTxRollbackError:
    def test_classifier_matches_no_tx_active_with_sqlite_error_code(self) -> None:
        exc = OperationalError("cannot rollback - no transaction is active", 1)
        assert _is_no_tx_rollback_error(exc) is True

    def test_classifier_matches_cannot_commit_wording(self) -> None:
        # The anchored "no transaction is active" substring covers both
        # commit and rollback wordings; bare "cannot rollback" is too permissive.
        exc = OperationalError("cannot commit - no transaction is active", 1)
        assert _is_no_tx_rollback_error(exc) is True

    def test_classifier_rejects_bare_cannot_rollback(self) -> None:
        # "cannot rollback" without the anchored "no transaction is active"
        # must NOT match, else unrelated errors trigger silent-swallow.
        exc = OperationalError("cannot rollback because the disk is full", 1)
        assert _is_no_tx_rollback_error(exc) is False

    def test_classifier_rejects_no_tx_message_with_wrong_code(self) -> None:
        # SQLITE_BUSY (5) with coincidental no-tx wording must NOT match —
        # the tx might be partially applied by a concurrent writer.
        exc = OperationalError("no transaction is active", 5)
        assert _is_no_tx_rollback_error(exc) is False

    def test_classifier_rejects_sqlite_error_with_unrelated_message(self) -> None:
        exc = OperationalError("disk full", 1)
        assert _is_no_tx_rollback_error(exc) is False

    def test_classifier_rejects_non_operational_error(self) -> None:
        # The check is OperationalError-only (others carry no SQLite code).
        from dqliteclient.exceptions import InterfaceError

        exc = InterfaceError("not connected")
        assert _is_no_tx_rollback_error(exc) is False


@pytest.mark.asyncio
class TestPoolResetNoTxPreservesSlot:
    async def test_reset_connection_preserves_slot_on_no_tx_rollback(self) -> None:
        """No-tx ROLLBACK error: _reset_connection returns True, flags scrubbed."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True
            conn._tx_owner = None

            async def fake_execute(sql: str) -> object:
                raise OperationalError("cannot rollback - no transaction is active", 1)

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert conn._in_transaction is False
        assert conn._tx_owner is None

    async def test_reset_connection_drops_slot_on_real_rollback_failure(self) -> None:
        """A real ROLLBACK failure (e.g. SQLITE_BUSY) still drops the slot."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True

            async def fake_execute(sql: str) -> object:
                raise OperationalError("database is locked", 5)

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is False

    async def test_reset_connection_drops_slot_on_unrelated_sqlite_error(
        self,
    ) -> None:
        """SQLITE_ERROR with a non-no-tx message still drops the slot."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True

            async def fake_execute(sql: str) -> object:
                raise OperationalError("syntax error near 'ROLLBACK'", 1)

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is False


class TestResetConnectionUmbrellaPredicate:
    """Pool issues ROLLBACK if ANY of _in_transaction, _savepoint_stack,
    or _savepoint_implicit_begin signals server-side state."""

    async def test_rollback_issued_when_only_savepoint_stack_set(self) -> None:
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = False
            conn._savepoint_stack = ["sp"]
            executed: list[str] = []

            async def fake_execute(sql: str) -> object:
                executed.append(sql)
                return None

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert executed == ["ROLLBACK"]
        assert conn._savepoint_stack == []

    async def test_rollback_issued_when_only_implicit_begin_set(self) -> None:
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = False
            conn._savepoint_implicit_begin = True
            executed: list[str] = []

            async def fake_execute(sql: str) -> object:
                executed.append(sql)
                return None

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert executed == ["ROLLBACK"]
        assert conn._savepoint_implicit_begin is False

    async def test_no_rollback_when_all_state_empty(self) -> None:
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = False
            conn._savepoint_stack = []
            conn._savepoint_implicit_begin = False
            executed: list[str] = []

            async def fake_execute(sql: str) -> object:
                executed.append(sql)
                return None

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert executed == []

    async def test_benign_no_tx_branch_clears_savepoint_state(self) -> None:
        """The no-tx branch clears savepoint stack and autobegin flag too."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True
            conn._savepoint_stack = ["sp"]
            conn._savepoint_implicit_begin = True

            async def fake_execute(sql: str) -> object:
                raise OperationalError("no transaction is active", 1)

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert conn._in_transaction is False
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
class TestUntrackedSavepointPredicate:
    """_has_untracked_savepoint participates in the pool-reset predicate:
    a bare quoted SAVEPOINT autobegins an untracked server-side tx that the
    other three flags miss, so the slot would leak a live tx without it."""

    async def test_quoted_savepoint_without_begin_sets_flag(self) -> None:
        """Bare quoted SAVEPOINT sets _has_untracked_savepoint; other flags stay clean."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False
        assert conn._in_transaction is False

    async def test_pool_reset_issues_rollback_on_untracked_flag(self) -> None:
        """Pool reset issues ROLLBACK when only _has_untracked_savepoint is set."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = False
            conn._savepoint_stack = []
            conn._savepoint_implicit_begin = False
            conn._has_untracked_savepoint = True
            executed: list[str] = []

            async def fake_execute(sql: str) -> object:
                executed.append(sql)
                return None

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert executed == ["ROLLBACK"]
        assert conn._has_untracked_savepoint is False

    async def test_quoted_savepoint_then_release_keeps_flag_sticky(self) -> None:
        """RELEASE can't be classified here (partial vs full); flag stays
        sticky until a definitive COMMIT/ROLLBACK/invalidate/close."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        conn._update_tx_flags_from_sql('RELEASE "Foo"')
        assert conn._has_untracked_savepoint is True

    async def test_commit_clears_untracked_flag(self) -> None:
        """A definitive COMMIT clears the flag."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        conn._in_transaction = True  # simulate the BEGIN side effect
        conn._update_tx_flags_from_sql("COMMIT")
        assert conn._has_untracked_savepoint is False

    async def test_rollback_clears_untracked_flag(self) -> None:
        """A definitive ROLLBACK clears the flag."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._has_untracked_savepoint = True
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._has_untracked_savepoint is False
        assert conn._in_transaction is False

    async def test_invalidate_clears_untracked_flag(self) -> None:
        """_invalidate clears the flag."""
        conn = DqliteConnection("localhost:9001")
        conn._has_untracked_savepoint = True
        conn._invalidate()
        assert conn._has_untracked_savepoint is False

    async def test_comment_prefixed_quoted_savepoint_sets_flag(self) -> None:
        """Comment stripping runs before the SAVEPOINT verb check, so a
        comment-prefixed quoted savepoint also sets the flag."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('/* annotation */ SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
