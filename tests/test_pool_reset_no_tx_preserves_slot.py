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

    def test_classifier_matches_cannot_rollback_wording(self) -> None:
        # The substring "cannot rollback" alone is also accepted, since
        # the C server has historically used both wordings.
        exc = OperationalError(1, "cannot rollback")
        assert _is_no_tx_rollback_error(exc) is True

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
