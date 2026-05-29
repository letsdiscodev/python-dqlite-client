"""__aexit__ hoists self.close() into an explicit Task with a done-callback before shielding.

A bare coro to asyncio.shield orphans its implicit Task on outer cancel, surfacing as
"Task exception was never retrieved" at GC if close() later raised.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_aexit_shield_uses_observer_not_bare_coro() -> None:
    """__aexit__ passes a Task/Future (not a bare coroutine) to asyncio.shield."""
    received_types: list[type] = []
    original_shield = asyncio.shield

    def recording_shield(arg, *args, **kwargs):
        received_types.append(type(arg))
        return original_shield(arg, *args, **kwargs)

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
    assert issubclass(arg_type, asyncio.Future) or arg_type is asyncio.Task, (
        f"__aexit__ passed {arg_type.__name__} to asyncio.shield — expected "
        "an explicit Task/Future hoist via asyncio.ensure_future. A bare "
        "coroutine here orphans the implicit Task that shield creates if "
        "the outer caller is cancelled mid-await."
    )
