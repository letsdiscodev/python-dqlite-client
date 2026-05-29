"""A failed reconnect must not leak the prior session's
``_invalidation_cause`` into the subsequent "Not connected" error."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import ClusterError, DqliteConnectionError

pytestmark = pytest.mark.asyncio


async def test_failed_connect_does_not_leak_prior_session_invalidation_cause() -> None:
    """After a session-1 invalidate then a session-2 connect failure, the
    cached ``_invalidation_cause`` must not carry the session-1 exception."""
    conn = DqliteConnection("localhost:9001")

    session_1_cause = RuntimeError("session-1 cause: leader-flip during query")
    conn._invalidate(session_1_cause)
    assert conn._invalidation_cause is session_1_cause

    async def _failing_dial(*args: object, **kwargs: object) -> object:
        raise OSError("simulated dial failure for session-2")

    # Patch the dial at the lowest level so the failure surfaces from
    # inside _connect_impl.
    with (
        patch.object(conn, "_dial_func", _failing_dial),
        pytest.raises((OSError, DqliteConnectionError, ClusterError)),
    ):
        await conn.connect()

    assert conn._invalidation_cause is not session_1_cause, (
        f"Stale session-1 _invalidation_cause leaked across the failed "
        f"connect attempt; got {conn._invalidation_cause!r}"
    )


async def test_ensure_connected_after_failed_connect_carries_no_stale_cause() -> None:
    """After the failed reconnect, the next op raises 'Not connected'
    without the session-1 cause on ``__cause__``."""
    conn = DqliteConnection("localhost:9001")
    session_1_cause = RuntimeError("session-1 cause")
    conn._invalidate(session_1_cause)

    async def _failing_dial(*args: object, **kwargs: object) -> object:
        raise OSError("dial fail")

    with (
        patch.object(conn, "_dial_func", _failing_dial),
        pytest.raises((OSError, DqliteConnectionError, ClusterError)),
    ):
        await conn.connect()

    with pytest.raises(DqliteConnectionError) as excinfo:
        conn._ensure_connected()
    assert excinfo.value.__cause__ is not session_1_cause, (
        f"Not-connected error carries stale session-1 cause on __cause__; "
        f"got {excinfo.value.__cause__!r}"
    )
