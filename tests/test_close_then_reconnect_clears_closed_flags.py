"""Pin: ``close()`` → ``connect()`` on the same ``DqliteConnection``
clears the sticky ``_closed`` / ``_closed_flag[0]`` markers so the
user-visible ``closed`` property reflects reality on the reconnected
slot.

The codebase supports close→reconnect cycles (inline rationale at
connection.py:1974-2000). The close path clears tx flags,
``_invalidation_cause``, and ``_bound_loop_ref`` for the reconnected
slot — but did NOT clear ``_closed`` / ``_closed_flag[0]``. As a
result:

1. ``conn.closed`` returned ``True`` on a fully-working reconnected
   connection — breaking the canonical ``if not conn.closed:`` idiom.
2. ``_closed_flag[0]`` stayed True so the GC-time
   ``_connection_unclosed_warning`` finalizer short-circuited on a
   reconnected-then-leaked socket, silently masking the leak.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


async def test_closed_property_reflects_state_after_simulated_reconnect() -> None:
    """Drive ``_connect_impl``'s success-path flag-clearing logic
    directly. The full network stack is replaced by an in-line
    inline-stub for ``_resolve_leader`` + ``DqliteProtocol`` — the
    success branch of ``_connect_impl`` is the only path the fix
    changes, and that branch reads no network state.

    Test contract: after the reconnect, ``closed`` is False AND
    ``_closed_flag[0]`` is False.
    """
    conn = DqliteConnection("localhost:9001")

    # Seed a closed state mirroring what close() leaves behind.
    conn._closed = True
    conn._closed_flag[0] = True
    conn._protocol = None
    conn._db_id = None

    # Patch the success-path internals so _connect_impl proceeds
    # without a live cluster. The flag-clearing site we are
    # pinning lives in the success branch AFTER open_database.
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_protocol = MagicMock()
    fake_protocol.handshake = AsyncMock(return_value=None)
    fake_protocol.open_database = AsyncMock(return_value=42)
    fake_protocol.close = MagicMock()

    async def _fake_open(*args: object, **kwargs: object) -> object:
        # Return (reader, writer) tuple shape the connect path expects.
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock(return_value=None)
        return reader, writer

    with (
        patch("dqliteclient.connection.open_connection", new=_fake_open),
        patch("dqliteclient.connection.DqliteProtocol", return_value=fake_protocol),
    ):
        await conn.connect()

    # CONTRACT: both flags must be cleared on the reconnected slot.
    # Without the fix, both stay True even though the connection
    # is fully working.
    assert conn.closed is False, (
        f"close() → connect() must reset .closed to False; got closed={conn.closed!r}"
    )
    assert conn._closed_flag[0] is False, (
        "close() → connect() must reset _closed_flag[0] so the "
        f"GC-time ResourceWarning re-arms; got {conn._closed_flag[0]!r}"
    )
    # Sanity: the rebuild actually succeeded.
    assert conn._protocol is fake_protocol
    assert conn._db_id == 42
