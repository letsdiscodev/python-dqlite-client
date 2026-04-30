"""Pin: ``DqliteConnection`` and ``ConnectionPool`` raise
``InterfaceError`` if used from a child process after ``os.fork``.

Fork-after-init is unsupported: the inherited TCP socket would be
shared with the parent (writes interleaving on the wire), and asyncio
primitives bound to the parent's loop are unusable in the child.

The fix records ``os.getpid()`` in ``__init__`` (both classes) and
``_check_in_use`` / ``acquire`` raise a clear ``InterfaceError``
("reconstruct from configuration in the target process") on pid
mismatch — symmetric with the existing ``__reduce__`` pickle guards.

The test does not need a live server: the pid check fires before any
network work; an unconnected instance is sufficient.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import ConnectionPool


def _run_in_child(check) -> bytes:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            try:
                check()
                os.write(w, b"NO_RAISE")
            except InterfaceError as e:
                msg = str(e)
                if "fork" in msg and "reconstruct from configuration" in msg:
                    os.write(w, b"OK")
                else:
                    os.write(w, f"WRONG_MSG:{msg}".encode())
            except Exception as e:  # noqa: BLE001
                os.write(w, f"WRONG_TYPE:{type(e).__name__}:{e}".encode())
            finally:
                os.close(w)
        finally:
            os._exit(0)
    os.close(w)
    result = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        result += chunk
    os.close(r)
    os.waitpid(pid, 0)
    return result


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_dqlite_connection_used_after_fork_raises_interface_error() -> None:
    conn = DqliteConnection("127.0.0.1:9999")
    assert conn._creator_pid == os.getpid()

    def child_check() -> None:
        # ``_check_in_use`` runs ``asyncio.get_running_loop`` after
        # the pid check. Drive a loop so the pid mismatch is the
        # raised error, not the "must be used from async context" one.
        async def run() -> None:
            conn._check_in_use()

        asyncio.run(run())

    result = _run_in_child(child_check)
    assert result == b"OK", f"child reported: {result!r}"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_connection_pool_acquire_after_fork_raises_interface_error() -> None:
    pool = ConnectionPool(addresses=["127.0.0.1:9999"])
    assert pool._creator_pid == os.getpid()

    def child_check() -> None:
        async def run() -> None:
            async with pool.acquire():
                pass

        asyncio.run(run())

    result = _run_in_child(child_check)
    assert result == b"OK", f"child reported: {result!r}"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_connection_pool_initialize_after_fork_raises_interface_error() -> None:
    pool = ConnectionPool(addresses=["127.0.0.1:9999"])

    def child_check() -> None:
        async def run() -> None:
            await pool.initialize()

        asyncio.run(run())

    result = _run_in_child(child_check)
    assert result == b"OK", f"child reported: {result!r}"


def test_dqlite_connection_close_after_fork_drops_inherited_state() -> None:
    """The fork short-circuit in ``DqliteConnection.close()`` must
    drop every reference that crosses the fork boundary —
    ``_pending_drain`` (parent-loop ``asyncio.Task``), transaction
    bookkeeping, savepoint stack, and ``_bound_loop`` — so GC in
    the child doesn't keep the inherited writer transport / socket
    FD alive via the Task's coroutine frame, and the child's
    ``in_transaction`` view stays self-consistent (False on a
    closed connection)."""
    conn = DqliteConnection("127.0.0.1:9999")
    # Stage state that would normally be cleared by _close_impl.
    sentinel_task = MagicMock(spec=asyncio.Task)
    conn._pending_drain = sentinel_task
    conn._in_transaction = True
    conn._tx_owner = MagicMock()
    conn._savepoint_stack.append("sp1")
    conn._savepoint_implicit_begin = True
    conn._has_untracked_savepoint = True
    conn._bound_loop = MagicMock()
    fake_parent_pid = conn._creator_pid + 1
    conn._creator_pid = fake_parent_pid

    async def run() -> None:
        with patch("dqliteclient.connection._current_pid", fake_parent_pid + 1):
            await conn.close()

    asyncio.run(run())

    assert conn._protocol is None
    assert conn._db_id is None
    assert conn._pending_drain is None
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    assert conn._has_untracked_savepoint is False
    assert conn._bound_loop is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_dqlite_connection_close_after_fork_short_circuits() -> None:
    """``close()`` in the child must not touch the inherited socket
    (which is shared with the parent — sending FIN would close it
    for the parent too). Short-circuits to a quiet local-state flip."""
    conn = DqliteConnection("127.0.0.1:9999")

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            try:

                async def run() -> None:
                    await conn.close()

                asyncio.run(run())
                # close() did not raise — the child's local state is
                # marked closed without touching the wire.
                os.write(w, b"OK")
            except Exception as e:  # noqa: BLE001
                os.write(w, f"WRONG:{type(e).__name__}:{e}".encode())
            finally:
                os.close(w)
        finally:
            os._exit(0)
    os.close(w)
    result = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        result += chunk
    os.close(r)
    os.waitpid(pid, 0)
    assert result == b"OK", f"child reported: {result!r}"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_connection_pool_close_after_fork_short_circuits() -> None:
    """``pool.close()`` in the child must not drain inherited connection
    FDs — those would close sockets the parent still uses. Short-
    circuits to a quiet local-state flip."""
    pool = ConnectionPool(addresses=["127.0.0.1:9999"])

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            try:

                async def run() -> None:
                    await pool.close()
                    assert pool._closed is True

                asyncio.run(run())
                os.write(w, b"OK")
            except Exception as e:  # noqa: BLE001
                os.write(w, f"WRONG:{type(e).__name__}:{e}".encode())
            finally:
                os.close(w)
        finally:
            os._exit(0)
    os.close(w)
    result = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        result += chunk
    os.close(r)
    os.waitpid(pid, 0)
    assert result == b"OK", f"child reported: {result!r}"
