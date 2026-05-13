"""Pin: ``DqliteConnection._connect_impl``'s ``except Exception: pass``
arm at ``connection.py:1351-1352`` swallows a non-Cancel Exception
raised while awaiting a prior ``_pending_drain`` task, so the
reconnect can proceed.

The adjacent arms are covered:
- ``pending is None`` / ``pending.done()`` — implicit on every fresh-
  instance test.
- ``CancelledError`` with outer-cancel delta vs. our-cancel delta —
  pinned by ``test_connect_cancelling_guard_branches.py`` and
  ``test_connection_reconnect_drain_slot.py``.

This file covers the fifth arm: the prior drain task surfaces a
non-Cancel ``Exception`` from ``await pending`` (e.g., a transport
RuntimeError synthesised by ``_invalidate``'s bounded
``wait_closed``). The arm must swallow it so the reconnect proceeds
to the dial.

A widening of the ``except Exception`` back to ``except
BaseException`` re-introduces the historical regression documented
in ``done/client-connect-pending-drain-suppress-baseexception-
consumes-task-cancel.md``; narrowing to e.g. ``except OSError``
lets a ``RuntimeError`` block reconnect. Either refactor would slip
past CI today; this test closes the regression trap.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import dqliteclient.connection as conn_mod
from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_connect_swallows_non_cancel_exception_from_prior_drain() -> None:
    """A prior drain task that, when cancelled, surfaces a non-Cancel
    Exception via ``await pending`` is swallowed by the
    ``except Exception: pass`` arm; ``_connect_impl`` proceeds to the
    dial. The fresh connect attempt's own dial-failure is what the
    caller sees, NOT the drain's failure."""
    conn = DqliteConnection("localhost:9001", timeout=1.0, close_timeout=1.0)

    # Drain task that, when cancelled, swallows the CancelledError and
    # re-raises RuntimeError. This is the production shape from an
    # ``_invalidate`` whose bounded ``wait_closed`` failed at the
    # transport layer — the cancel arrives, the bounded wait surfaces
    # an underlying transport error.
    started = asyncio.Event()

    async def drain_raising_runtime() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Surface a non-Cancel Exception from inside the drain
            # body so ``await pending`` sees a RuntimeError, not a
            # CancelledError.
            raise RuntimeError("synthetic drain transport failure") from None

    prior = asyncio.get_running_loop().create_task(drain_raising_runtime())
    await started.wait()
    conn._pending_drain = prior

    # Drive ``connect()`` past the drain block and into the dial; make
    # the dial fail promptly with a different, distinguishable error
    # so we can assert the user sees THIS error, not the drain's
    # RuntimeError. The pattern follows
    # ``test_connection_reconnect_drain_slot.py``.
    real_proto = conn_mod.DqliteProtocol  # type: ignore[attr-defined]
    real_open = asyncio.open_connection

    async def fake_open_connection(host: str, port: int, **_kwargs: object):
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
            raise RuntimeError("dial-failure-marker")

        async def wait_closed(self) -> None:
            return None

        def close(self) -> None:
            return None

    async def fake_abort(self: object) -> None:
        return None

    conn_mod.DqliteProtocol = _StubProtocol  # type: ignore[assignment,attr-defined]
    asyncio.open_connection = fake_open_connection  # type: ignore[assignment]
    original_abort = DqliteConnection._abort_protocol
    DqliteConnection._abort_protocol = fake_abort

    try:
        # The caller sees the dial-failure-marker — confirming
        # _connect_impl reached the dial. The drain's RuntimeError
        # was swallowed by the ``except Exception: pass`` arm.
        with pytest.raises(RuntimeError, match="dial-failure-marker"):
            await conn.connect()
    finally:
        conn_mod.DqliteProtocol = real_proto  # type: ignore[attr-defined]
        asyncio.open_connection = real_open
        DqliteConnection._abort_protocol = original_abort

    # Drain slot was cleared after the swallowing arm — line 1353.
    assert conn._pending_drain is None
    # The prior task is done (we cancelled it).
    assert prior.done()
    # And it really did raise the non-Cancel Exception (not just get
    # cancelled). Awaiting it again surfaces the RuntimeError.
    with pytest.raises(RuntimeError, match="synthetic drain transport failure"):
        await prior
