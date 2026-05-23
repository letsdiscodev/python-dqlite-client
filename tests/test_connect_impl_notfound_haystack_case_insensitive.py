"""Pin: ``DqliteConnection._connect_impl``'s SQLITE_NOTFOUND
leader-flip classifier lower-cases the haystack before comparing
against ``LEADER_LOST_DB_LOOKUP_SUBSTRING`` — symmetric with the
``_run_protocol`` sibling hardened in the previous round.

The previous round lowercased the haystack at ``_run_protocol``
(around line 2622-2632) so a future upstream wording variant like
"No database opened" (capital N) keeps classifying as
leader-flip. The handshake-time site at ``_connect_impl`` did NOT
get the symmetric treatment; this pin closes the gap.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import dqliteclient.connection as conn_mod
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqlitewire import SQLITE_NOTFOUND


@pytest.mark.parametrize(
    "raw_message",
    [
        "No database opened (db_id=7)",  # capital N — future upstream drift
        "NO DATABASE OPENED (db_id=7)",  # all-caps
        "No Database Opened (db_id=7)",  # title-case
    ],
)
@pytest.mark.asyncio
async def test_connect_impl_classifies_capitalised_leader_lost_wording(
    monkeypatch: pytest.MonkeyPatch,
    raw_message: str,
) -> None:
    """Capitalised variants of the upstream wording must still
    rewrap as ``DqliteConnectionError`` so SA's ``is_disconnect``
    invalidates the pool slot. Pre-fix: the raw startswith without
    ``.lower()`` silently drops the rewrap for any capitalisation
    change.
    """

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
                raw_message,
                code=SQLITE_NOTFOUND,
                raw_message=raw_message,
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
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        DqliteConnection._abort_protocol = real_abort
