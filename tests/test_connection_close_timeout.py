"""``DqliteConnection.close`` must bound the transport drain.

The local side of the socket is already closed after ``writer.close()``;
awaiting ``wait_closed()`` is best-effort cleanup, not
correctness-critical. An unresponsive peer must not be able to stall
``engine.dispose()`` or SIGTERM shutdown by refusing to acknowledge
the FIN.

Peer of ISSUE-295.
"""

from __future__ import annotations

import asyncio
import math
import time
import weakref
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


class _HungProtocol:
    """A protocol whose ``wait_closed`` never resolves."""

    def __init__(self) -> None:
        self.close = MagicMock()
        self._waited = asyncio.Event()

    async def wait_closed(self) -> None:
        # Await indefinitely; the bounded close in DqliteConnection
        # should cancel us via asyncio.wait_for.
        self._waited.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_close_bounds_wait_closed_with_close_timeout() -> None:
    """A hung peer must not be able to stall ``close()`` indefinitely."""
    conn = DqliteConnection("localhost:9001", close_timeout=0.1)
    hung = _HungProtocol()
    conn._protocol = hung  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    t0 = time.monotonic()
    await conn.close()
    elapsed = time.monotonic() - t0

    hung.close.assert_called_once()
    # Allow generous slack; assertion is that it RETURNED bounded.
    assert elapsed < 1.0, f"close() should complete within ~close_timeout, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_close_happy_path_completes_promptly() -> None:
    """A well-behaved peer resolves wait_closed immediately; close()
    completes well under the close_timeout budget.
    """
    conn = DqliteConnection("localhost:9001", close_timeout=5.0)

    class _PromptProtocol:
        def __init__(self) -> None:
            self.close = MagicMock()

        async def wait_closed(self) -> None:
            return None

    conn._protocol = _PromptProtocol()  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    t0 = time.monotonic()
    await conn.close()
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"happy-path close() should be near-instant, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_close_oserror_during_drain_is_swallowed() -> None:
    """OSError on wait_closed is normal on an already-closed writer;
    the caller sees a successful close.
    """
    conn = DqliteConnection("localhost:9001")

    class _OSErroringProtocol:
        def __init__(self) -> None:
            self.close = MagicMock()

        async def wait_closed(self) -> None:
            raise OSError("socket already closed")

    conn._protocol = _OSErroringProtocol()  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    await conn.close()  # Must not raise.


@pytest.mark.asyncio
async def test_close_timeout_error_during_drain_is_swallowed() -> None:
    """``TimeoutError`` raised by the protocol's ``wait_closed`` (slow
    peer that never ACKs the FIN beyond the ``asyncio.wait_for``
    budget) is caught via the single ``OSError`` entry in the except
    clause — regression guard for the tuple narrowing.
    """
    conn = DqliteConnection("localhost:9001")

    class _TimeoutProtocol:
        def __init__(self) -> None:
            self.close = MagicMock()

        async def wait_closed(self) -> None:
            raise TimeoutError("slow peer")

    conn._protocol = _TimeoutProtocol()  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    await conn.close()  # Must not raise.


@pytest.mark.asyncio
async def test_abort_protocol_timeout_error_during_drain_is_swallowed() -> None:
    """Mirror of the close() TimeoutError pin for the _abort_protocol
    path — both share the OSError-narrowed except clause.
    """
    conn = DqliteConnection("localhost:9001", close_timeout=0.1)

    class _TimeoutProtocol:
        def __init__(self) -> None:
            self.close = MagicMock()

        async def wait_closed(self) -> None:
            raise TimeoutError("slow peer")

    conn._protocol = _TimeoutProtocol()  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    await conn._abort_protocol()  # Must not raise.


@pytest.mark.asyncio
async def test_close_cancellederror_escapes() -> None:
    """Cancellation from an outer ``asyncio.timeout`` / task cancel
    must propagate out of close() — structured concurrency requires
    it. Swallowing CancelledError inside wait_closed would leak the
    cancel signal.
    """
    conn = DqliteConnection("localhost:9001")

    class _CancellingProtocol:
        def __init__(self) -> None:
            self.close = MagicMock()

        async def wait_closed(self) -> None:
            raise asyncio.CancelledError

    conn._protocol = _CancellingProtocol()  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    with pytest.raises(asyncio.CancelledError):
        await conn.close()


def test_close_timeout_must_be_positive_finite() -> None:
    """Reject zero / negative / NaN / inf to mirror ``timeout`` validation."""
    with pytest.raises(ValueError, match="close_timeout"):
        DqliteConnection("localhost:9001", close_timeout=0.0)
    with pytest.raises(ValueError, match="close_timeout"):
        DqliteConnection("localhost:9001", close_timeout=-1.0)
    with pytest.raises(ValueError, match="close_timeout"):
        DqliteConnection("localhost:9001", close_timeout=math.inf)
    with pytest.raises(ValueError, match="close_timeout"):
        DqliteConnection("localhost:9001", close_timeout=math.nan)


@pytest.mark.asyncio
async def test_abort_protocol_also_uses_close_timeout() -> None:
    """The connect-failure path shares the same bounded drain."""
    conn = DqliteConnection("localhost:9001", close_timeout=0.1)
    hung = _HungProtocol()
    conn._protocol = hung  # type: ignore[assignment]
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())

    t0 = time.monotonic()
    await conn._abort_protocol()
    elapsed = time.monotonic() - t0

    hung.close.assert_called_once()
    assert elapsed < 1.0, f"_abort_protocol() should be bounded, took {elapsed:.3f}s"
