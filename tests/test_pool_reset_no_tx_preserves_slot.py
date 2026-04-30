"""Pool reset preserves the slot when ROLLBACK reports "no tx active".

When a leader-flip cascade lands on many connections, the new leader
has no record of any in-flight transaction, so the next ROLLBACK
issued by the pool's ``_reset_connection`` replies with the
deterministic ``SQLITE_ERROR`` + "no transaction is active" pair.
The connection itself is healthy (TCP-alive, protocol-clean); the
local ``_in_transaction=True`` was the only thing out-of-sync, and
the failed ROLLBACK just confirmed the server's view.

Treating every ROLLBACK exception as "drop the slot" wastes a
connect round-trip per slot under that cascade. Pin the desired
behaviour: the no-tx case preserves the slot and scrubs the local
flags; any other failure still drops the slot.

Mirrors the discrimination already done by ``transaction()`` at
``connection.py`` (the no-tx-active swallow), now centralised via
``_is_no_tx_rollback_error``.
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
        exc = OperationalError(1, "cannot rollback - no transaction is active")
        assert _is_no_tx_rollback_error(exc) is True

    def test_classifier_matches_cannot_commit_wording(self) -> None:
        # Both upstream wordings contain "no transaction is active"
        # so the anchored substring covers commit and rollback
        # equally. The bare "cannot rollback" token was previously
        # also accepted but was too permissive.
        exc = OperationalError(1, "cannot commit - no transaction is active")
        assert _is_no_tx_rollback_error(exc) is True

    def test_classifier_rejects_bare_cannot_rollback(self) -> None:
        # A message containing "cannot rollback" without the anchored
        # "no transaction is active" clause must NOT match — any
        # unrelated SQLite (or DQLITE_ERROR=1) error happening to
        # contain those words would otherwise trigger silent-swallow.
        exc = OperationalError(1, "cannot rollback because the disk is full")
        assert _is_no_tx_rollback_error(exc) is False

    def test_classifier_rejects_no_tx_message_with_wrong_code(self) -> None:
        # SQLITE_BUSY (5) primary code with a coincidental no-tx wording
        # must NOT match — the user's tx might have been partially
        # applied by a concurrent writer and we cannot silently swallow.
        exc = OperationalError(5, "no transaction is active")
        assert _is_no_tx_rollback_error(exc) is False

    def test_classifier_rejects_sqlite_error_with_unrelated_message(self) -> None:
        exc = OperationalError(1, "disk full")
        assert _is_no_tx_rollback_error(exc) is False

    def test_classifier_rejects_non_operational_error(self) -> None:
        # InterfaceError, ProtocolError, etc. carry no SQLite code; the
        # check is OperationalError-only.
        from dqliteclient.exceptions import InterfaceError

        exc = InterfaceError("not connected")
        assert _is_no_tx_rollback_error(exc) is False


@pytest.mark.asyncio
class TestPoolResetNoTxPreservesSlot:
    async def test_reset_connection_preserves_slot_on_no_tx_rollback(self) -> None:
        """ROLLBACK that reports the deterministic no-tx error must
        leave _reset_connection returning True, with flags scrubbed."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True
            # Leave _tx_owner None to mirror the raw-BEGIN path that
            # most often hits this surface (pool's reset runs from the
            # release task, not the owning one).
            conn._tx_owner = None

            async def fake_execute(sql: str) -> object:
                # Simulate the server's deterministic reply.
                raise OperationalError(1, "cannot rollback - no transaction is active")

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert conn._in_transaction is False
        assert conn._tx_owner is None

    async def test_reset_connection_drops_slot_on_real_rollback_failure(self) -> None:
        """A real ROLLBACK failure (e.g. SQLITE_BUSY) must still drop
        the slot — preserving it would leak server-side state."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True

            async def fake_execute(sql: str) -> object:
                raise OperationalError(5, "database is locked")

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is False

    async def test_reset_connection_drops_slot_on_unrelated_sqlite_error(
        self,
    ) -> None:
        """SQLITE_ERROR with a non-no-tx message must still drop the
        slot — only the deterministic no-tx wording is benign."""
        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
        conn = DqliteConnection("localhost:9001")
        with (
            patch.object(DqliteConnection, "is_connected", new=True),
            patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        ):
            conn._in_transaction = True

            async def fake_execute(sql: str) -> object:
                raise OperationalError(1, "syntax error near 'ROLLBACK'")

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is False


class TestResetConnectionUmbrellaPredicate:
    """Pin the umbrella defensive ROLLBACK predicate: the pool must
    issue ROLLBACK if ANY of _in_transaction, _savepoint_stack, or
    _savepoint_implicit_begin signals server-side state that would
    poison the next acquirer. The strict _in_transaction-only check
    missed cases like quoted-name SAVEPOINTs that the parser
    deliberately does not push (case-sensitivity trade-off)."""

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
        assert executed == []  # No ROLLBACK issued — clean state.

    async def test_benign_no_tx_branch_clears_savepoint_state(self) -> None:
        """The 'server already auto-rolled back' branch must clear
        the savepoint stack and autobegin flag along with the existing
        _in_transaction / _tx_owner clears."""
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
                raise OperationalError(1, "no transaction is active")

            with patch.object(conn, "execute", new=fake_execute):
                result = await pool._reset_connection(conn)

        assert result is True
        assert conn._in_transaction is False
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
class TestUntrackedSavepointPredicate:
    """Pin: ``_has_untracked_savepoint`` participates in the pool-reset
    predicate. A bare ``SAVEPOINT "Foo"`` issued without a preceding
    BEGIN auto-begins a server-side transaction that the local stack
    deliberately does NOT track (``_parse_savepoint_name`` returns None
    for quoted identifiers per the case-sensitivity trade-off). Without
    the flag, all three other predicate fields stay False/empty and the
    pool returns the slot with a live tx — leaking across acquirers."""

    async def test_quoted_savepoint_without_begin_sets_flag(self) -> None:
        """Pin the wire-up: bare ``SAVEPOINT "Foo"`` must set
        ``_has_untracked_savepoint=True``. All three other flags stay
        False/empty by design."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        # Other flags stay False — case-sensitivity trade-off preserved.
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False
        assert conn._in_transaction is False

    async def test_pool_reset_issues_rollback_on_untracked_flag(self) -> None:
        """Pool reset must issue ROLLBACK when only
        ``_has_untracked_savepoint`` is set."""
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
        """``RELEASE "Foo"`` cannot be classified at this layer: it
        might be a partial release (server still holds outer untracked
        frames) or a full release. The flag stays sticky until a
        definitive close (COMMIT/ROLLBACK on the outer tx, or
        invalidate/close of the connection)."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        conn._update_tx_flags_from_sql('RELEASE "Foo"')
        # Flag remains True — pool reset will fire the safety ROLLBACK.
        assert conn._has_untracked_savepoint is True

    async def test_commit_clears_untracked_flag(self) -> None:
        """A definitive COMMIT clears the flag — the autobegun tx is
        gone, no leak surface remains."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        # The user does an explicit BEGIN+COMMIT to end the tx after
        # confusion. The COMMIT branch in the parser unconditionally
        # clears the untracked flag (the only autobegun tx is gone).
        conn._in_transaction = True  # simulate the BEGIN side effect
        conn._update_tx_flags_from_sql("COMMIT")
        assert conn._has_untracked_savepoint is False

    async def test_rollback_clears_untracked_flag(self) -> None:
        """A definitive ROLLBACK clears the flag (no active tx and no
        untracked savepoint frames remain server-side)."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._has_untracked_savepoint = True
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._has_untracked_savepoint is False
        assert conn._in_transaction is False

    async def test_invalidate_clears_untracked_flag(self) -> None:
        """``_invalidate`` participates in the all-clear discipline."""
        conn = DqliteConnection("localhost:9001")
        conn._has_untracked_savepoint = True
        conn._invalidate()
        assert conn._has_untracked_savepoint is False

    async def test_comment_prefixed_quoted_savepoint_sets_flag(self) -> None:
        """Comment stripping must apply before the SAVEPOINT verb
        check, so a comment-prefixed quoted savepoint also sets the
        flag — pin the parity with the parser's leading-comment path."""
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql('/* annotation */ SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
