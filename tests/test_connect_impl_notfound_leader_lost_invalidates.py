"""``_connect_impl``'s ``_is_leader_flip`` covers the substring-gated SQLITE_NOTFOUND arm
during OPEN (sibling of the ``_run_protocol`` site).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import dqliteclient.connection as conn_mod
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqlitewire import LEADER_LOST_DB_LOOKUP_SUBSTRING, SQLITE_NOTFOUND


@pytest.mark.asyncio
async def test_connect_translates_notfound_leader_lost_to_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leader-lost SQLITE_NOTFOUND from open_database rewraps as DqliteConnectionError
    and invalidates the protocol."""

    async def _fake_open_connection(
        host: str, port: int, **_kwargs: object
    ) -> tuple[object, object]:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    class _StubProtocol:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._client_id = 0
            self._writer = args[1] if len(args) >= 2 else MagicMock()

        async def handshake(self) -> None:
            return None

        async def open_database(self, _name: str) -> int:
            raise OperationalError(
                f"{LEADER_LOST_DB_LOOKUP_SUBSTRING} (db_id=7)",
                code=SQLITE_NOTFOUND,
                raw_message=f"{LEADER_LOST_DB_LOOKUP_SUBSTRING} (db_id=7)",
            )

        async def wait_closed(self) -> None:
            return None

        def close(self) -> None:
            return None

    async def _fake_abort(self: DqliteConnection) -> None:
        self._protocol = None

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_abort = DqliteConnection._abort_protocol

    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    monkeypatch.setattr("asyncio.open_connection", _fake_open_connection)
    DqliteConnection._abort_protocol = _fake_abort

    try:
        conn = DqliteConnection("127.0.0.1:9001", timeout=2.0)
        with pytest.raises(DqliteConnectionError) as ei:
            await conn.connect()
        assert ei.value.code == SQLITE_NOTFOUND
        assert "no longer leader" in str(ei.value)
        assert conn._protocol is None
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = real_abort


@pytest.mark.asyncio
async def test_connect_propagates_notfound_lookup_stmt_as_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative twin: a LOOKUP_STMT SQLITE_NOTFOUND is a server bug, not a flip; it must
    propagate as-is rather than rewrap as DqliteConnectionError."""

    async def _fake_open_connection(
        host: str, port: int, **_kwargs: object
    ) -> tuple[object, object]:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    class _StubProtocol:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._client_id = 0
            self._writer = args[1] if len(args) >= 2 else MagicMock()

        async def handshake(self) -> None:
            return None

        async def open_database(self, _name: str) -> int:
            raise OperationalError(
                "no statement with the given id 7",
                code=SQLITE_NOTFOUND,
                raw_message="no statement with the given id 7",
            )

        async def wait_closed(self) -> None:
            return None

        def close(self) -> None:
            return None

    async def _fake_abort(self: DqliteConnection) -> None:
        self._protocol = None

    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_abort = DqliteConnection._abort_protocol
    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    monkeypatch.setattr("asyncio.open_connection", _fake_open_connection)
    DqliteConnection._abort_protocol = _fake_abort

    try:
        conn = DqliteConnection("127.0.0.1:9001", timeout=2.0)
        with pytest.raises(OperationalError) as ei:
            await conn.connect()
        assert ei.value.code == SQLITE_NOTFOUND
        assert not isinstance(ei.value, DqliteConnectionError)
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = real_abort
