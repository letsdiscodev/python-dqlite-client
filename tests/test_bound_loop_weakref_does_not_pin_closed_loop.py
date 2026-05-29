"""_bound_loop_ref is a weakref so a closed loop is not pinned in memory.

Once the bound loop is closed and otherwise unreferenced, _check_in_use raises
InterfaceError rather than silently rebinding (which would mask cross-loop misuse).
"""

from __future__ import annotations

import asyncio
import contextlib
import weakref

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def test_bound_loop_ref_is_weakref_type() -> None:
    """The slot must be a weakref.ref."""
    conn = DqliteConnection("127.0.0.1:1", timeout=0.5)

    async def _bind() -> None:
        with contextlib.suppress(InterfaceError):
            conn._check_in_use()

    asyncio.run(_bind())
    if conn._bound_loop_ref is not None:
        assert isinstance(conn._bound_loop_ref, weakref.ref), (
            f"_bound_loop_ref must be weakref.ref, got {type(conn._bound_loop_ref)}"
        )


@pytest.mark.asyncio
async def test_check_in_use_raises_when_bound_loop_was_closed() -> None:
    """A dead _bound_loop_ref weakref makes _check_in_use raise, not silently rebind."""
    import os as _os

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pool_released = False
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._creator_pid = _os.getpid()

    # A weakref to a dead object returns None, simulating a GC'd loop.
    class _LoopProxy:
        pass

    proxy = _LoopProxy()
    ref = weakref.ref(proxy)
    del proxy
    assert ref() is None

    # _check_in_use only does an identity comparison, so any weakref-able stand-in works.
    conn._bound_loop_ref = ref  # type: ignore[assignment]

    with pytest.raises(InterfaceError, match="closed event loop"):
        conn._check_in_use()
