"""``_check_in_use``'s ``_in_use=True`` arm reports both the claimant task
(holder of ``_in_use``) and the contender task that hit the guard."""

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
    """``_in_use=True`` raises InterfaceError with both claimant and contender reprs."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._in_use = True
    conn._closed = False
    conn._creator_pid = os.getpid()
    conn._pool_released = False
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
    conn._in_transaction = False
    conn._tx_owner = None
    # MagicMock's repr embeds 'MagicMock', which the assertion sentinel-checks.
    claimant_task = MagicMock(spec=asyncio.Task)
    conn._in_use_claimant = claimant_task

    with pytest.raises(InterfaceError, match="another operation is in progress") as ei:
        conn._check_in_use()
    msg = str(ei.value)
    assert "claimant:" in msg, f"expected claimant: prefix; got {msg}"
    assert "MagicMock" in msg or "Mock" in msg, (
        f"expected MagicMock-flavour claimant repr; got {msg}"
    )
    assert "current task:" in msg, f"expected current task prefix; got {msg}"


@pytest.mark.asyncio
async def test_check_in_use_with_no_claimant_repr_reports_none() -> None:
    """``_in_use=True`` without a stashed claimant still raises, with claimant=None."""
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
