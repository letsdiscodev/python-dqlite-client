"""When _run_protocol sees ``OperationalError(code=SQLITE_NOTFOUND,
raw_message="no database opened ...")``, the connection MUST be
invalidated — Go-parity with ``driverError`` ``errNotFound →
ErrBadConn`` "potentially after leadership loss".

The orthogonal ``LOOKUP_STMT`` arm of upstream ``gateway.c`` emits
the same primary code (12 = SQLITE_NOTFOUND) but with a different
message ("no statement with the given id ..."). That arm is a
server-side state bug rather than a transport flip; invalidating
would over-trigger and break the existing PEP 249 ``InternalError``
classification (the dbapi maps SQLITE_NOTFOUND to ``InternalError``,
matching stdlib ``sqlite3``). The discriminator is therefore
substring-based (the wire-side constant
``LEADER_LOST_DB_LOOKUP_SUBSTRING``), not code-based.
"""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError
from dqlitewire import LEADER_LOST_DB_LOOKUP_SUBSTRING, SQLITE_NOTFOUND


@pytest.mark.asyncio
async def test_notfound_lookup_db_invalidates_connection() -> None:
    """``code == SQLITE_NOTFOUND`` AND ``raw_message`` begins with
    ``LEADER_LOST_DB_LOOKUP_SUBSTRING`` → invalidate. This is the
    leader-flip arm of ``gateway.c::LOOKUP_DB``."""
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

    # Connection must be invalidated — go-dqlite's ErrBadConn parity.
    assert conn._protocol is None, (
        "code=SQLITE_NOTFOUND with 'no database opened' message must "
        "invalidate the connection (Go errNotFound → ErrBadConn arm)"
    )


@pytest.mark.asyncio
async def test_notfound_lookup_stmt_does_not_invalidate_connection() -> None:
    """``code == SQLITE_NOTFOUND`` with a different message (the
    ``LOOKUP_STMT`` arm of upstream ``gateway.c``) must NOT invalidate.
    The PEP 249 classifier still maps to ``InternalError`` —
    orthogonal to the invalidation decision here."""
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

    # Stmt-id-bug arm: connection stays healthy.
    assert conn._protocol is not None, (
        "code=SQLITE_NOTFOUND with 'no statement with the given id' "
        "message is a server-side state bug, not a leader flip — "
        "must NOT invalidate the connection"
    )


@pytest.mark.asyncio
async def test_notfound_with_empty_raw_message_does_not_invalidate() -> None:
    """Defensive: a synthetic OperationalError with empty
    ``raw_message`` (e.g. from a test or a poorly-constructed mock)
    must NOT invalidate — the substring check on an empty string
    returns False with ``startswith``."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol: object, db_id: int) -> None:
        raise OperationalError("synthetic", code=SQLITE_NOTFOUND)

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None
