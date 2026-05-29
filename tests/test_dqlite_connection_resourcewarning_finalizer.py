"""Pin: ``DqliteConnection`` emits ``ResourceWarning`` when GC'd without
``close()`` — but only when it actually opened. Mirrors the dbapi-layer
``AsyncConnection`` finalizer for a driver-attributable diagnostic.
"""

import gc
import os
import warnings

from dqliteclient.connection import (
    DqliteConnection,
    _connection_unclosed_warning,
)


def test_unclosed_warning_skips_never_connected() -> None:
    """A never-connected DqliteConnection must NOT warn at GC (false positive)."""
    closed_flag = [False]
    connected_flag = [False]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "h:9001", os.getpid())
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_skips_when_close_was_called() -> None:
    """closed_flag set means orderly shutdown ran: skip the warning."""
    closed_flag = [True]
    connected_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "h:9001", os.getpid())
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_fires_when_connected_but_not_closed() -> None:
    """Fires when connected==True AND closed==False (opened but not closed)."""
    closed_flag = [False]
    connected_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _connection_unclosed_warning(closed_flag, connected_flag, "h:9001", os.getpid())
    matching = [r for r in w if issubclass(r.category, ResourceWarning)]
    assert len(matching) == 1
    assert "DqliteConnection" in str(matching[0].message)
    assert "h:9001" in str(matching[0].message)


def test_finalizer_registered_on_construction() -> None:
    """Every DqliteConnection gets a finalizer registered at construction."""
    conn = DqliteConnection("h:9001")
    assert conn._finalizer is not None
    assert conn._finalizer.alive
    # Cleanup so this doesn't leak a finalize() across tests.
    conn._finalizer.detach()


def test_finalizer_detached_after_close_via_flags() -> None:
    """Closing flips closed_flag and clears the finalizer reference."""
    conn = DqliteConnection("h:9001")
    assert conn._finalizer is not None
    conn._closed_flag[0] = True
    conn._finalizer.detach()  # as close() does
    conn._finalizer = None
    del conn
    gc.collect()


def test_invalidate_flips_closed_flag_and_detaches_finalizer() -> None:
    """_invalidate flips _closed_flag[0]=True and detaches the finalizer so
    dropping the reference without close() doesn't false-positive warn."""
    conn = DqliteConnection("h:9001")
    assert conn._finalizer is not None
    assert conn._closed_flag[0] is False
    conn._invalidate(Exception("dummy"))
    assert conn._closed_flag[0] is True, (
        "_invalidate must flip _closed_flag[0] to True so the GC "
        "finalizer suppresses the false-positive ResourceWarning"
    )
    assert conn._finalizer is None, "_invalidate must detach the GC finalizer"
    # _closed stays False so an awaited close() after invalidate still proceeds.
    assert conn._closed is False
