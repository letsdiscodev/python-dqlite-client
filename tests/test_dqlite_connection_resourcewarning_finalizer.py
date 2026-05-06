"""Pin: ``DqliteConnection`` emits ``ResourceWarning`` when GC'd
without ``close()`` — but only when the connection actually opened.

Mirrors the dbapi-layer ``AsyncConnection`` finalizer so direct
dqliteclient consumers (sqlalchemy-dqlite, third-party adopters)
get a driver-attributable diagnostic instead of asyncio's
generic "Task was destroyed but it is pending" pointing at the
wrong layer.
"""

import gc
import warnings

from dqliteclient.connection import (
    DqliteConnection,
    _connection_unclosed_warning,
)


def test_unclosed_warning_skips_never_connected() -> None:
    """A never-connected DqliteConnection must NOT emit the
    warning at GC — that would be a false positive (early-error /
    test-fixture flow)."""
    closed_flag = [False]
    connected_flag = [False]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "h:9001")
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_skips_when_close_was_called() -> None:
    """closed_flag set means orderly shutdown ran. Skip the warning."""
    closed_flag = [True]
    connected_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "h:9001")
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_fires_when_connected_but_not_closed() -> None:
    """The two-flag gate fires when connected==True AND
    closed==False — the user opened the transport but forgot to
    close()."""
    closed_flag = [False]
    connected_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "h:9001")
    matching = [r for r in w if issubclass(r.category, ResourceWarning)]
    assert len(matching) == 1
    assert "DqliteConnection" in str(matching[0].message)
    assert "h:9001" in str(matching[0].message)


def test_finalizer_registered_on_construction() -> None:
    """Pin: every DqliteConnection instance gets a registered
    finalizer at construction. Drop the reference and confirm the
    finalize was wired (it would be cleared on close())."""
    conn = DqliteConnection("h:9001")
    assert conn._finalizer is not None
    assert conn._finalizer.alive
    # Explicit cleanup so this doesn't leak a finalize() across tests.
    conn._finalizer.detach()


def test_finalizer_detached_after_close_via_flags() -> None:
    """Closing flips closed_flag and clears the finalizer reference."""
    conn = DqliteConnection("h:9001")
    assert conn._finalizer is not None
    conn._closed_flag[0] = True
    # Simulate detach as close() does.
    conn._finalizer.detach()
    conn._finalizer = None
    del conn
    gc.collect()
    # No warning — closed_flag was set, finalizer was detached.


def test_invalidate_flips_closed_flag_and_detaches_finalizer() -> None:
    """``_invalidate`` must flip ``_closed_flag[0] = True`` and detach
    the GC finalizer — the transport is gone after invalidate so a
    user dropping the reference WITHOUT ``await close()`` should not
    see a false-positive ``ResourceWarning`` from the finalizer.
    Mirrors ``close()``'s end-state for the GC-readable flag."""
    conn = DqliteConnection("h:9001")
    assert conn._finalizer is not None
    assert conn._closed_flag[0] is False
    # _invalidate doesn't need a real protocol; just call it.
    conn._invalidate(Exception("dummy"))
    assert conn._closed_flag[0] is True, (
        "_invalidate must flip _closed_flag[0] to True so the GC "
        "finalizer suppresses the false-positive ResourceWarning"
    )
    assert conn._finalizer is None, "_invalidate must detach the GC finalizer"
    # Sanity: a subsequent _close_impl is still a no-op idempotent path
    # (the explicit-close flag _closed is intentionally NOT set by
    # invalidate, so an awaited close() after invalidate proceeds).
    assert conn._closed is False
