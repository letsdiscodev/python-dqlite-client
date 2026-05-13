"""Pin: ``_pool_unclosed_warning`` finalizer is fork-safe — it
skips the ResourceWarning emission when invoked in a forked child
whose parent owned the captured ``closed_flag`` / ``reserved_flag``
snapshots.

Mirrors the dbapi sibling pin at
``python-dqlite-dbapi/tests/aio/test_async_unclosed_warning_skips_in_fork_child.py``
and the connection-side pin at
``tests/test_connection_unclosed_finalizer_fork_safe.py``.

Without the pid guard, a parent that constructed a
``ConnectionPool``, called ``initialize()`` (flipping
``reserved_flag[0] = True``) and forked, would have the child emit
a false-positive "pool was garbage-collected" warning. AND the
inherited queued ``DqliteConnection`` siblings would each emit
their own false positive, for an N+1 cascade.
"""

from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient.pool import _pool_unclosed_warning


def test_pool_finalizer_short_circuits_on_pid_mismatch_no_warning() -> None:
    """In a simulated forked child, ``_pool_unclosed_warning`` must
    not emit a ResourceWarning even though ``reserved_flag=True`` and
    ``closed_flag=False`` would otherwise trigger one."""
    closed_flag = [False]
    reserved_flag = [True]
    parent_pid = os.getpid()
    child_pid = parent_pid + 1

    with (
        patch("dqliteclient.pool.get_current_pid", return_value=child_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag, None, parent_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert not leak_warnings, (
        f"forked-child pool finalizer must not emit ResourceWarning; got "
        f"{[str(w.message) for w in leak_warnings]}"
    )


def test_pool_finalizer_emits_warning_when_pid_matches() -> None:
    """Negative pin: in-process, the warning still fires when
    reserved_flag=True and closed_flag=False."""
    closed_flag = [False]
    reserved_flag = [True]
    same_pid = os.getpid()

    with (
        patch("dqliteclient.pool.get_current_pid", return_value=same_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag, None, same_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert len(leak_warnings) == 1, (
        f"in-process pool finalizer must emit one ResourceWarning when "
        f"closed_flag=False and reserved_flag=True; got "
        f"{[str(w.message) for w in leak_warnings]}"
    )


@pytest.mark.asyncio
async def test_pool_finalizer_pid_mismatch_skips_even_with_nonempty_queue() -> None:
    """Forked child path: even when the queue carries entries
    (parent state), the child must not emit the warning."""
    closed_flag = [False]
    reserved_flag = [True]
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
    for _ in range(3):
        queue.put_nowait(object())
    parent_pid = os.getpid()
    child_pid = parent_pid + 1

    with (
        patch("dqliteclient.pool.get_current_pid", return_value=child_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag, queue, parent_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert not leak_warnings


def test_pool_finalizer_registration_captures_creator_pid() -> None:
    """Pin: the pool's finalize registration must include the
    creator_pid so the pid guard has a parent-pid baseline. Future
    refactors must not silently drop the arg and revert the fix."""
    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    finalizer = pool._finalizer
    assert finalizer is not None
    try:
        peeked = finalizer.peek()
        assert peeked is not None
        _obj, _func, args, _kwargs = peeked
        assert pool._creator_pid in args, (
            "pool finalizer registration dropped the creator_pid arg — "
            "fork-pid guard cannot fire"
        )
    finally:
        finalizer.detach()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_pool_finalizer_in_forked_child_does_not_emit_warning() -> None:
    """End-to-end pin: construct a ConnectionPool, simulate the
    reserved_flag flip, fork, GC the inherited pool in the child,
    and verify the child does not emit a false-positive
    ResourceWarning to its stderr."""
    import contextlib as _contextlib
    import gc
    import sys

    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool(addresses=["127.0.0.1:9999"], min_size=0, max_size=2)
    pool._reserved_flag[0] = True
    assert pool._finalizer is not None

    pipe_r, pipe_w = os.pipe()
    err_r, err_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(pipe_r)
        os.close(err_r)
        os.dup2(err_w, sys.stderr.fileno())
        os.close(err_w)
        with _contextlib.suppress(BaseException):
            del pool
            gc.collect()
        os.write(pipe_w, b"DONE")
        os.close(pipe_w)
        os._exit(0)

    os.close(pipe_w)
    os.close(err_w)
    while os.read(pipe_r, 4096):
        pass
    os.close(pipe_r)
    child_stderr = b""
    while True:
        chunk = os.read(err_r, 4096)
        if not chunk:
            break
        child_stderr += chunk
    os.close(err_r)
    os.waitpid(pid, 0)
    decoded = child_stderr.decode(errors="replace")
    assert "ConnectionPool was garbage-collected" not in decoded, (
        f"forked-child pool finalizer emitted false-positive ResourceWarning: {decoded!r}"
    )
    del pool
