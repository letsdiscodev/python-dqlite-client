"""Pin: a failed reconnect attempt does NOT leak the prior session's
``_invalidation_cause`` into the subsequent "Not connected" error.

Sequence:
1. ``connect()`` succeeds.
2. Live op fails → ``_invalidate(X)`` → ``_protocol = None``,
   ``_invalidation_cause = X``.
3. ``connect()`` is retried (allowed because ``_protocol is None``).
4. The new connect FAILS mid-handshake.
5. The caller's next op raises ``DqliteConnectionError("Not connected")``;
   that error's ``__cause__`` must NOT point at the session-1 ``X``
   (which would mislead forensic recovery).

The morning patch cleared ``_invalidation_cause`` on success
(connection.py:1616) and on close (connection.py:2000); the failure
paths in ``_connect_impl`` (the four ``_abort_protocol`` arms) did
not. The fix is a single ``self._invalidation_cause = None`` at the
top of ``_connect_impl``, semantically: "starting a new attempt; the
old cause is no longer relevant".
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import ClusterError, DqliteConnectionError

pytestmark = pytest.mark.asyncio


async def test_failed_connect_does_not_leak_prior_session_invalidation_cause() -> None:
    """After a session-1 invalidate followed by a session-2 connect
    failure, the cached ``_invalidation_cause`` must NOT carry the
    session-1 exception. Without the top-of-method clear, the next
    ``_ensure_connected`` call surfaces ``DqliteConnectionError(...)
    from X`` where X is the OLD session-1 cause.
    """
    conn = DqliteConnection("localhost:9001")

    # Simulate session-1 invalidation:
    session_1_cause = RuntimeError("session-1 cause: leader-flip during query")
    conn._invalidate(session_1_cause)
    assert conn._invalidation_cause is session_1_cause

    # Make the next connect fail at the dial step.
    async def _failing_dial(*args: object, **kwargs: object) -> object:
        raise OSError("simulated dial failure for session-2")

    # The connect path runs through find_leader / _resolve_leader
    # / DqliteConnection.connect; patch the dial at the lowest level
    # so the failure surfaces from inside _connect_impl.
    with (
        patch.object(conn, "_dial_func", _failing_dial),
        pytest.raises((OSError, DqliteConnectionError, ClusterError)),
    ):
        await conn.connect()

    # CONTRACT: the stale session-1 cause must be cleared. The fix at
    # the top of _connect_impl clears it; if the fix is missing, the
    # field still points at session_1_cause and the next error chain
    # leaks the wrong cause.
    assert conn._invalidation_cause is not session_1_cause, (
        f"Stale session-1 _invalidation_cause leaked across the failed "
        f"connect attempt; got {conn._invalidation_cause!r}"
    )


async def test_ensure_connected_after_failed_connect_carries_no_stale_cause() -> None:
    """End-to-end: after the failed reconnect, the next op raises
    'Not connected' WITHOUT the session-1 cause on ``__cause__``.
    """
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

    # Next op: _ensure_connected reads _invalidation_cause.
    with pytest.raises(DqliteConnectionError) as excinfo:
        conn._ensure_connected()
    # __cause__ should NOT be the session-1 exception.
    assert excinfo.value.__cause__ is not session_1_cause, (
        f"Not-connected error carries stale session-1 cause on __cause__; "
        f"got {excinfo.value.__cause__!r}"
    )
