"""SQLITE_NOTFOUND with the "no database opened" message invalidates (Go errNotFound
-> ErrBadConn parity); the LOOKUP_STMT arm shares the code but a different message and
is a server-side bug, so the discriminator is substring-based, not code-based."""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError
from dqlitewire import LEADER_LOST_DB_LOOKUP_SUBSTRING, SQLITE_NOTFOUND


@pytest.mark.asyncio
async def test_notfound_lookup_db_invalidates_connection() -> None:
    """SQLITE_NOTFOUND + raw_message starting with the leader-loss substring invalidates."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol: object, db_id: int) -> None:
        raise OperationalError(
            f"{LEADER_LOST_DB_LOOKUP_SUBSTRING} (db_id={db_id})",
            code=SQLITE_NOTFOUND,
            raw_message=f"{LEADER_LOST_DB_LOOKUP_SUBSTRING} (db_id={db_id})",
        )

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is None, (
        "code=SQLITE_NOTFOUND with 'no database opened' message must "
        "invalidate the connection (Go errNotFound → ErrBadConn arm)"
    )


@pytest.mark.asyncio
async def test_notfound_lookup_stmt_does_not_invalidate_connection() -> None:
    """SQLITE_NOTFOUND with the LOOKUP_STMT message must NOT invalidate."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol: object, db_id: int) -> None:
        raise OperationalError(
            "no statement with the given id 7",
            code=SQLITE_NOTFOUND,
            raw_message="no statement with the given id 7",
        )

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None, (
        "code=SQLITE_NOTFOUND with 'no statement with the given id' "
        "message is a server-side state bug, not a leader flip — "
        "must NOT invalidate the connection"
    )


@pytest.mark.asyncio
async def test_notfound_with_empty_raw_message_does_not_invalidate() -> None:
    """An empty ``raw_message`` must NOT invalidate (startswith on "" is False)."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol: object, db_id: int) -> None:
        raise OperationalError("synthetic", code=SQLITE_NOTFOUND)

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None
