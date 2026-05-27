"""Pin: ``DqliteConnection.__aexit__`` hoists ``self.close()`` into
an explicit Task with a done-callback observer BEFORE shielding so
an outer cancel landing mid-await does not orphan an implicit Task
``asyncio.shield`` would otherwise create.

The orphan would surface as ``Task exception was never retrieved``
at GC if ``close()`` later raised an Exception with no awaiter
holding the future. Same regression class as the prior dbapi/sa
shield-bare-coro sweep — the client-side `__aexit__` site was
missed by that sweep's scoping.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_aexit_shield_uses_observer_not_bare_coro() -> None:
    """Verify ``__aexit__`` schedules ``self.close()`` via
    ``asyncio.ensure_future`` + ``add_done_callback`` instead of
    passing a bare coroutine to ``asyncio.shield``.

    We patch ``asyncio.shield`` to record what it received: an
    ``asyncio.Task`` (good — explicit hoist) or a bare coroutine
    (bad — implicit Task orphan risk).
    """
    received_types: list[type] = []
    original_shield = asyncio.shield

    def recording_shield(arg, *args, **kwargs):
        received_types.append(type(arg))
        return original_shield(arg, *args, **kwargs)

    # Build a stub DqliteConnection-like object so we don't need a
    # live cluster connection. Just import the class and replace
    # ``close`` with an awaitable that completes immediately.
    from dqliteclient.connection import DqliteConnection

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._closed_flag = [False]
    conn._connected_flag = [False]
    conn._creator_pid = 0

    async def fake_close() -> None:
        return None

    conn.close = fake_close

    with patch("asyncio.shield", side_effect=recording_shield):
        await DqliteConnection.__aexit__(conn, None, None, None)

    assert received_types, "asyncio.shield was not invoked"
    arg_type = received_types[0]
    # The hoist produces a ``Task`` (or ``Future``). A bare-coro
    # call to ``shield`` would record a ``coroutine`` type instead.
    assert issubclass(arg_type, asyncio.Future) or arg_type is asyncio.Task, (
        f"__aexit__ passed {arg_type.__name__} to asyncio.shield — expected "
        "an explicit Task/Future hoist via asyncio.ensure_future. A bare "
        "coroutine here orphans the implicit Task that shield creates if "
        "the outer caller is cancelled mid-await."
    )
