"""``DqliteConnection._run_protocol`` schedules a best-effort
INTERRUPT on a FRESH socket when CancelledError lands mid-operation,
mirroring go-dqlite's ``Rows.Close`` → ``Protocol.Interrupt``
discipline (``protocol.go:79-116``). The fire-and-forget background
task releases server-side resources held by the cancelled query
(Raft / WAL / memory) without holding up the cancel propagation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dqliteclient.connection import (
    DqliteConnection,
    _send_interrupt_on_fresh_socket,
)


@pytest.mark.asyncio
async def test_send_interrupt_on_fresh_socket_best_effort_swallows_dial_failure() -> None:
    """The helper swallows OSError from a refused dial — server-side
    cleanup is best-effort and a failed interrupt does not propagate."""

    async def refusing_open(*_args: object, **_kwargs: object):
        raise ConnectionRefusedError("simulated")

    with patch("dqliteclient._dial.open_connection_with_keepalive", refusing_open):
        # Should not raise.
        await _send_interrupt_on_fresh_socket(
            "localhost:9001",
            db_id=1,
            dial_timeout=0.1,
            interrupt_timeout=0.5,
        )


@pytest.mark.asyncio
async def test_send_interrupt_on_fresh_socket_swallows_timeout() -> None:
    """A slow dial that exceeds the interrupt budget is swallowed."""

    async def slow_open(*_args: object, **_kwargs: object):
        await asyncio.sleep(2.0)
        raise AssertionError("should have timed out")

    with patch("dqliteclient._dial.open_connection_with_keepalive", slow_open):
        await _send_interrupt_on_fresh_socket(
            "localhost:9001",
            db_id=1,
            dial_timeout=0.05,
            interrupt_timeout=0.1,
        )


@pytest.mark.asyncio
async def test_run_protocol_schedules_interrupt_on_cancel() -> None:
    """When _run_protocol catches CancelledError, it schedules a
    fire-and-forget interrupt task on a fresh socket. The task is
    named ``dqlite-interrupt-<address>`` for log correlation."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    # Pretend the connection is established.
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = 7
    conn._invalidation_cause = None

    # Make _ensure_connected return our fake protocol.
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 7))

    # The fn passed to _run_protocol raises CancelledError.
    async def cancelled_op(_p: object, _db: int) -> None:
        raise asyncio.CancelledError()

    interrupt_called = asyncio.Event()

    async def fake_interrupt(*_args: object, **_kwargs: object) -> None:
        interrupt_called.set()

    with patch(
        "dqliteclient.connection._send_interrupt_on_fresh_socket",
        fake_interrupt,
    ):
        with pytest.raises(asyncio.CancelledError):
            await conn._run_protocol(cancelled_op)

        # Yield to let the fire-and-forget task run.
        await asyncio.wait_for(interrupt_called.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_run_protocol_skips_interrupt_when_db_id_is_none() -> None:
    """No INTERRUPT is sent when ``_db_id`` is ``None`` (the connection
    never reached open_database)."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    fake_protocol = MagicMock()
    conn._protocol = fake_protocol
    conn._db_id = None
    conn._invalidation_cause = None
    conn._ensure_connected = MagicMock(return_value=(fake_protocol, 0))  # noqa: SLF001  # type: ignore[method-assign]

    async def cancelled_op(_p: object, _db: int) -> None:
        raise asyncio.CancelledError()

    interrupt_called = False

    async def fake_interrupt(*_args: object, **_kwargs: object) -> None:
        nonlocal interrupt_called
        interrupt_called = True

    with patch(
        "dqliteclient.connection._send_interrupt_on_fresh_socket",
        fake_interrupt,
    ):
        with pytest.raises(asyncio.CancelledError):
            await conn._run_protocol(cancelled_op)
        # Brief yield so any spurious task would have run.
        await asyncio.sleep(0)

    assert not interrupt_called
