"""Connection/protocol error-propagation defensive paths: non-leader
OperationalError propagates as-is (vs leader-change wrapping), _abort_protocol
no-protocol early return, and unexpected mid-ROWS EmptyResponse -> ProtocolError."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages.responses import EmptyResponse


class TestConnectReraisesNonLeaderOperationalError:
    async def test_non_leader_operational_error_propagates_as_is(
        self, monkeypatch: pytest.MonkeyPatch, cluster_address: str = "localhost:9001"
    ) -> None:
        """Leader-change codes wrap to DqliteConnectionError; a non-leader code
        (SQLITE_CORRUPT=11) propagates as the original OperationalError."""

        async def _raise_corrupt(self: DqliteProtocol) -> None:
            raise OperationalError("database disk image is malformed", 11)

        monkeypatch.setattr(DqliteProtocol, "handshake", _raise_corrupt)

        conn = DqliteConnection(cluster_address, timeout=2.0)
        with pytest.raises(OperationalError) as excinfo:
            await conn.connect()
        assert excinfo.value.code == 11
        assert "malformed" in str(excinfo.value)
        # connect() must abort the protocol so no socket leaks.
        assert conn._protocol is None


class TestAbortProtocolNoProtocolEarlyReturn:
    async def test_abort_on_never_connected_returns_early(self) -> None:
        """_abort_protocol early-returns when no protocol exists yet."""
        conn = DqliteConnection("localhost:19001", timeout=2.0)
        assert conn._protocol is None
        await conn._abort_protocol()
        assert conn._protocol is None


class TestReadContinuationRejectsUnexpectedEmptyResponse:
    async def test_emptyresponse_mid_rows_continuation_raises_protocol_error(
        self,
    ) -> None:
        """An EmptyResponse mid-ROWS (INTERRUPT ack the client never sent) must
        raise ProtocolError rather than deadlock awaiting a continuation."""
        proto = DqliteProtocol.__new__(DqliteProtocol)
        decoder = MagicMock()
        decoder.decode_continuation = MagicMock(return_value=EmptyResponse())
        proto._decoder = decoder
        proto._address = "localhost:19001"
        proto._timeout = 2.0
        # _read_data must not be called on this path; fail loudly if it is.
        proto._read_data = AsyncMock(
            side_effect=AssertionError(
                "_read_data must not be called when decoder yields EmptyResponse"
            )
        )

        with pytest.raises(ProtocolError, match="Unexpected EmptyResponse"):
            await proto._read_continuation(deadline=999_999.0)
