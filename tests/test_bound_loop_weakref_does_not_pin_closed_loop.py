"""Pin: ``DqliteConnection._bound_loop_ref`` is a weakref so a
long-lived connection whose loop has been closed (but the connection
itself was not properly ``close()``d) does NOT pin the loop in memory.

The dbapi/aio layer uses the same weakref discipline; the client
layer was a strong reference, leaving closed loops reachable for
direct ``DqliteConnection`` users via the loop's selector / finalize
state and any other loop-attached objects.

Pin two contracts:
1. Once the bound loop is closed AND released by all other references,
   ``_check_in_use`` raises ``InterfaceError("bound to a closed event
   loop")`` — must NOT silently rebind to a fresh loop, that would
   mask cross-loop misuse.
2. ``_bound_loop_ref`` is a ``weakref.ref`` (not the loop directly).
"""

from __future__ import annotations

import asyncio
import contextlib
import weakref

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def test_bound_loop_ref_is_weakref_type() -> None:
    """Static-shape pin: the slot must be a weakref.ref."""
    conn = DqliteConnection("127.0.0.1:1", timeout=0.5)

    async def _bind() -> None:
        # Trigger _check_in_use to lazily bind the loop.
        with contextlib.suppress(InterfaceError):
            conn._check_in_use()

    asyncio.run(_bind())
    # After binding, the slot is a weakref.ref (or None if some other
    # path cleared it; tolerate that).
    if conn._bound_loop_ref is not None:
        assert isinstance(conn._bound_loop_ref, weakref.ref), (
            f"_bound_loop_ref must be weakref.ref, got {type(conn._bound_loop_ref)}"
        )


@pytest.mark.asyncio
async def test_check_in_use_raises_when_bound_loop_was_closed() -> None:
    """Construct a conn whose ``_bound_loop_ref`` is a dead weakref
    (the loop it pointed to has been GC'd). ``_check_in_use`` must
    raise ``InterfaceError`` with a "closed event loop" message —
    NOT silently rebind to the running loop, which would mask
    cross-loop misuse."""
    import os as _os

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pool_released = False
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._creator_pid = _os.getpid()

    # Build a weakref to a temporary that we'll let die — the
    # weakref returns None, simulating the loop having been GC'd.
    class _LoopProxy:  # cheap object that supports weakref
        pass

    proxy = _LoopProxy()
    ref = weakref.ref(proxy)
    del proxy  # weakref now returns None
    assert ref() is None

    # Type-stub the assignment: _bound_loop_ref is annotated as
    # weakref.ref[AbstractEventLoop] | None; the test stand-in is
    # functionally identical (any object with weakref support works
    # because _check_in_use only does an identity comparison).
    conn._bound_loop_ref = ref  # type: ignore[assignment]

    with pytest.raises(InterfaceError, match="closed event loop"):
        conn._check_in_use()
