"""Pin connection / protocol error propagation defensive paths
reported as uncovered by ``pytest --cov``.

Lines covered (pre-pragma):

- connection.py:408 — ``connect()`` re-raises ``OperationalError``
  whose code is NOT in ``_LEADER_ERROR_CODES`` (leader-change
  errors translate to ``DqliteConnectionError`` for the pool's
  reconnect-elsewhere flow; non-leader errors propagate as-is).
- connection.py:531 — ``_abort_protocol`` no-protocol early
  return.
- protocol.py:600 — ``_read_continuation`` raises
  ``ProtocolError`` on an unexpected ``EmptyResponse``
  mid-ROWS-stream (server-side INTERRUPT acknowledgement that
  the client never sent).

The L408 path is the differentiator between transport-failure
(driver-internal retry) and SQL-failure (caller's exception
handler) — a regression that swallowed it would let SQL errors
masquerade as transport failures. The L600 path is the only place
that distinguishes a corrupt mid-stream EmptyResponse from a
legitimate continuation frame; without it the protocol would
deadlock on the corrupt input.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages.responses import EmptyResponse

# ---------------------------------------------------------------------------
# connection.py:408 — re-raise non-leader-change OperationalError
# ---------------------------------------------------------------------------


class TestConnectReraisesNonLeaderOperationalError:
    async def test_non_leader_operational_error_propagates_as_is(
        self, monkeypatch: pytest.MonkeyPatch, cluster_address: str = "localhost:19001"
    ) -> None:
        """``connect()`` translates leader-change OperationalErrors
        to ``DqliteConnectionError`` (so the pool's reconnect-
        elsewhere flow fires); a non-leader code propagates as the
        original ``OperationalError``. SQLITE_CORRUPT (code=11) is
        a deterministic non-leader code — pin the propagate path."""

        # Patch handshake to raise after the TCP connect succeeds.
        async def _raise_corrupt(self: DqliteProtocol) -> None:
            raise OperationalError(11, "database disk image is malformed")

        monkeypatch.setattr(DqliteProtocol, "handshake", _raise_corrupt)

        conn = DqliteConnection(cluster_address, timeout=2.0)
        with pytest.raises(OperationalError) as excinfo:
            await conn.connect()
        # Must surface as the original OperationalError (not wrapped
        # to DqliteConnectionError); code preserved.
        assert excinfo.value.code == 11
        assert "malformed" in str(excinfo.value)
        # connect() must have called _abort_protocol() so no socket
        # leaks; the protocol slot is None after the failure.
        assert conn._protocol is None


# ---------------------------------------------------------------------------
# connection.py:531 — _abort_protocol no-protocol early return
# ---------------------------------------------------------------------------


class TestAbortProtocolNoProtocolEarlyReturn:
    async def test_abort_on_never_connected_returns_early(self) -> None:
        """``_abort_protocol`` is normally called from connect()'s
        failure paths after ``self._protocol`` was assigned. The
        early-return guards against a caller that invokes it before
        any protocol exists. Drives connection.py:531."""
        conn = DqliteConnection("localhost:19001", timeout=2.0)
        assert conn._protocol is None
        # Must succeed without raising and without dereferencing
        # the None protocol.
        await conn._abort_protocol()
        assert conn._protocol is None


# ---------------------------------------------------------------------------
# protocol.py:600 — Unexpected EmptyResponse during ROWS continuation
# ---------------------------------------------------------------------------


class TestReadContinuationRejectsUnexpectedEmptyResponse:
    async def test_emptyresponse_mid_rows_continuation_raises_protocol_error(
        self,
    ) -> None:
        """The ROWS continuation reader treats an
        ``EmptyResponse`` as a server-side INTERRUPT
        acknowledgement. Since the client-layer ``query_sql`` flow
        never sends INTERRUPT, an ``EmptyResponse`` here means the
        server cancelled out-of-band (or a corrupt/MitM-modified
        stream). Surface as ``ProtocolError`` so the protocol does
        not deadlock waiting for a continuation that will never
        come. Drives protocol.py:600."""
        # Build a Protocol with a mocked decoder; bypass __init__ so
        # we don't need real reader/writer streams.
        proto = DqliteProtocol.__new__(DqliteProtocol)
        decoder = MagicMock()
        decoder.decode_continuation = MagicMock(return_value=EmptyResponse())
        proto._decoder = decoder
        proto._address = "localhost:19001"
        proto._timeout = 2.0
        # _read_data should never be called on this code path; if it
        # is, fail loudly rather than silently.
        proto._read_data = AsyncMock(
            side_effect=AssertionError(
                "_read_data must not be called when decoder yields EmptyResponse"
            )
        )

        with pytest.raises(ProtocolError, match="Unexpected EmptyResponse"):
            await proto._read_continuation(deadline=999_999.0)
