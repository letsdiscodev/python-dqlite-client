"""Pin: ``_connection_unclosed_warning`` finalizer is fork-safe — it
skips the ResourceWarning emission when invoked in a forked child
whose parent owned the captured ``closed_flag`` / ``connected_flag``
snapshots.

Mirrors the dbapi sibling pin at
``python-dqlite-dbapi/tests/aio/test_async_unclosed_warning_skips_in_fork_child.py``.
Without the pid guard, a parent that constructed a
``DqliteConnection``, called ``connect()`` (flipping
``connected_flag[0] = True``), and forked, would have the child
emit a false-positive ResourceWarning at GC time — under
``pytest -W error::ResourceWarning`` that crashes the child's test.
"""

from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from dqliteclient.connection import _connection_unclosed_warning


def test_connection_finalizer_short_circuits_on_pid_mismatch_no_warning() -> None:
    """In a simulated forked child, ``_connection_unclosed_warning`` must
    not emit a ResourceWarning even though ``connected_flag=True`` and
    ``closed_flag=False`` would otherwise trigger one. The flags are
    a parent snapshot frozen at fork-time; the parent may very well
    close after the fork."""
    closed_flag = [False]
    connected_flag = [True]
    parent_pid = os.getpid()
    child_pid = parent_pid + 1  # simulated child pid

    with (
        patch("dqliteclient.connection.get_current_pid", return_value=child_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "host:9001", parent_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert not leak_warnings, (
        f"forked-child connection finalizer must not emit ResourceWarning; got "
        f"{[str(w.message) for w in leak_warnings]}"
    )


def test_connection_finalizer_emits_warning_when_pid_matches() -> None:
    """Negative pin: in the SAME process that registered the
    finalizer (the normal GC path) and with the flags indicating
    a leak, the warning still fires. Behaviour on the non-forked
    path is unchanged."""
    closed_flag = [False]
    connected_flag = [True]
    same_pid = os.getpid()

    with (
        patch("dqliteclient.connection.get_current_pid", return_value=same_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "host:9001", same_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert len(leak_warnings) == 1, (
        f"in-process finalizer must emit one ResourceWarning when "
        f"closed_flag=False and connected_flag=True; got "
        f"{[str(w.message) for w in leak_warnings]}"
    )


def test_connection_finalizer_skips_when_closed_flag_set() -> None:
    """Pre-existing gate is preserved: explicit close
    (closed_flag=True) skips the warning."""
    closed_flag = [True]
    connected_flag = [True]
    same_pid = os.getpid()

    with (
        patch("dqliteclient.connection.get_current_pid", return_value=same_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "host:9001", same_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert not leak_warnings


def test_connection_finalizer_skips_when_never_connected() -> None:
    """Pre-existing gate is preserved: never-connected
    (connected_flag=False) skips the warning."""
    closed_flag = [False]
    connected_flag = [False]
    same_pid = os.getpid()

    with (
        patch("dqliteclient.connection.get_current_pid", return_value=same_pid),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "host:9001", same_pid)

    leak_warnings = [w for w in captured if issubclass(w.category, ResourceWarning)]
    assert not leak_warnings


def test_connection_finalizer_registration_captures_creator_pid() -> None:
    """The finalizer registration must include the creator pid so
    the pid check has a parent-pid baseline to compare against.
    Pin so a future refactor doesn't silently drop the arg."""
    from dqliteclient.connection import DqliteConnection

    conn = DqliteConnection("h:9001")
    finalizer = conn._finalizer
    assert finalizer is not None
    try:
        peeked = finalizer.peek()
        assert peeked is not None
        _obj, _func, args, _kwargs = peeked
        # The creator pid is the fourth positional arg
        # (closed_flag, connected_flag, address, creator_pid).
        assert conn._creator_pid in args, (
            "connection finalizer registration dropped the creator_pid arg — "
            "fork-pid guard cannot fire"
        )
    finally:
        finalizer.detach()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_connection_finalizer_in_forked_child_does_not_emit_warning() -> None:
    """End-to-end pin: construct a DqliteConnection (which registers
    a finalizer with creator_pid), simulate the connected_flag flip,
    fork, GC the inherited DqliteConnection in the child, and verify
    the child does not emit a false-positive ResourceWarning to its
    stderr.

    Mirrors the dbapi-side pin at
    ``test_async_unclosed_warning_skips_in_fork_child.py::test_async_finalizer_in_forked_child_does_not_emit_warning``.
    """
    import contextlib as _contextlib
    import gc
    import sys

    from dqliteclient.connection import DqliteConnection

    conn = DqliteConnection("127.0.0.1:9999")
    # Flip connected_flag manually so the finalizer's two pre-existing
    # gates would otherwise fire if not for the new pid gate.
    conn._connected_flag[0] = True
    assert conn._finalizer is not None

    pipe_r, pipe_w = os.pipe()
    err_r, err_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        # Child: redirect stderr to the pipe so we can inspect what
        # the finalizer emitted (warnings go to stderr by default).
        os.close(pipe_r)
        os.close(err_r)
        os.dup2(err_w, sys.stderr.fileno())
        os.close(err_w)
        with _contextlib.suppress(BaseException):
            del conn
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
    assert "was garbage-collected" not in decoded, (
        f"forked-child connection finalizer emitted false-positive ResourceWarning: {decoded!r}"
    )
    del conn
