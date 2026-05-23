"""Pin: ``DqliteConnection._check_in_use``'s ``_in_use=True`` arm
reports both the claimant task (the task that holds ``_in_use``)
and the contender task (the task that just hit the guard).

Pre-fix the arm reported only the contender's repr; operators
reading "another operation is in progress (current task: <X>)"
parsed X as the task in progress, but X was the BLOCKED task,
not the blocking one. Mirrors the ``_tx_owner`` arm's
owner+current shape.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_check_in_use_reports_claimant_repr() -> None:
    """When ``_in_use=True``, the raised InterfaceError must
    contain the repr of the claimant task AND the contender task."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._in_use = True
    conn._closed = False
    conn._creator_pid = os.getpid()
    conn._pool_released = False
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
    conn._in_transaction = False
    conn._tx_owner = None
    # Simulate a claimant task that holds _in_use. A bare MagicMock's
    # repr() naturally embeds 'MagicMock' so we sentinel-check via that.
    claimant_task = MagicMock(spec=asyncio.Task)
    conn._in_use_claimant = claimant_task

    with pytest.raises(InterfaceError, match="another operation is in progress") as ei:
        conn._check_in_use()
    msg = str(ei.value)
    # Both reprs must appear.
    assert "claimant:" in msg, f"expected claimant: prefix; got {msg}"
    assert "MagicMock" in msg or "Mock" in msg, (
        f"expected MagicMock-flavour claimant repr; got {msg}"
    )
    assert "current task:" in msg, f"expected current task prefix; got {msg}"


@pytest.mark.asyncio
async def test_check_in_use_with_no_claimant_repr_reports_none() -> None:
    """Defensive: if ``_in_use=True`` was set without stashing a
    claimant (legacy fixture / partial-init path), the diagnostic
    still surfaces but with claimant=None."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._in_use = True
    conn._in_use_claimant = None
    conn._closed = False
    conn._creator_pid = os.getpid()
    conn._pool_released = False
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
    conn._in_transaction = False
    conn._tx_owner = None
    with pytest.raises(InterfaceError, match="claimant: None") as ei:
        conn._check_in_use()
    assert "current task:" in str(ei.value)
