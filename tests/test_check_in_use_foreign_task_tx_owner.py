"""A sibling task using a connection whose tx another task owns raises InterfaceError."""

import asyncio
import contextlib

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_foreign_task_during_active_tx_raises_interface_error() -> None:
    """Task A holds a tx; task B's use must raise InterfaceError citing both tasks."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    conn._in_transaction = True
    fake_owner = asyncio.create_task(asyncio.sleep(60))
    conn._tx_owner = fake_owner

    try:
        with pytest.raises(InterfaceError) as excinfo:
            conn._check_in_use()
        msg = str(excinfo.value)
        assert "owned by another task" in msg
        assert "owner" in msg
        assert "current" in msg
    finally:
        fake_owner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fake_owner


@pytest.mark.asyncio
async def test_same_task_during_active_tx_passes() -> None:
    """The owner task may keep using the connection; only foreign tasks are rejected."""
    conn = DqliteConnection("localhost:9001", timeout=2.0)
    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._check_in_use()
