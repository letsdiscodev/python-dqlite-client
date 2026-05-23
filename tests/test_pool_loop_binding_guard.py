"""Pin: ``ConnectionPool`` lazy-binds to an event loop on first use
and raises ``InterfaceError`` on cross-loop misuse — mirroring the
sibling discipline at ``DqliteConnection._check_in_use``.

The pool's ``asyncio.Queue`` and ``asyncio.Lock`` are constructed
eagerly in ``__init__`` and lazy-bind to whichever loop touches them
first. Pre-fix, silent cross-loop misuse (pool created on loop A,
awaited from loop B) surfaced only as a deep asyncio-internal error
or a deadlock. Post-fix, ``_check_loop_binding`` at each public
entry point produces an actionable ``InterfaceError``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import ConnectionPool


def test_pool_construction_outside_loop_does_not_bind() -> None:
    """Factory-style construction outside any running loop must work;
    the bind happens lazily on first use."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=1, timeout=1.0)
    assert pool._loop_ref is None, "construction outside a running loop must not eagerly bind"


@pytest.mark.asyncio
async def test_pool_first_use_lazy_binds_to_current_loop() -> None:
    """First call to a public method records the bound loop."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=1, timeout=1.0)
    pool._check_loop_binding()
    assert pool._loop_ref is not None
    assert pool._loop_ref() is asyncio.get_running_loop()


def test_pool_cross_loop_misuse_raises_interface_error() -> None:
    """A pool bound to loop A and awaited from loop B must raise
    ``InterfaceError("bound to a different event loop")`` instead of
    a deep asyncio-internal error."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=1, timeout=1.0)

    # Bind to loop A on this thread.
    loop_a = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(asyncio.sleep(0))  # establish the loop
        # Stand in for a real call by invoking the guard directly on loop A.
        loop_a.run_until_complete(_bind_on_loop(pool))
    finally:
        loop_a.close()

    # Now drive the guard from a fresh loop B on a different thread.
    captured: list[BaseException] = []

    def _runner() -> None:
        loop_b = asyncio.new_event_loop()
        try:

            async def _drive() -> None:
                try:
                    pool._check_loop_binding()
                except BaseException as exc:
                    captured.append(exc)

            loop_b.run_until_complete(_drive())
        finally:
            loop_b.close()

    t = threading.Thread(target=_runner)
    t.start()
    t.join(timeout=2.0)

    assert captured, "guard did not raise from loop B"
    exc = captured[0]
    assert isinstance(exc, InterfaceError), f"expected InterfaceError, got {exc!r}"
    assert "different event loop" in str(exc), f"expected the cross-loop diagnostic; got {exc!s}"


async def _bind_on_loop(pool: ConnectionPool) -> None:
    pool._check_loop_binding()


def test_pool_guard_outside_async_context_raises_interface_error() -> None:
    """Calling _check_loop_binding outside any running loop must raise
    a clean ``InterfaceError("async context")`` — the same shape the
    connection-layer guard raises in the same scenario."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=1, timeout=1.0)
    with pytest.raises(InterfaceError, match="async context"):
        pool._check_loop_binding()


def test_pool_guard_after_closed_loop_raises_clean_diagnostic() -> None:
    """If the bound loop is GC'd (the weakref expires), a subsequent
    call from a fresh loop must raise ``InterfaceError("closed event
    loop")`` rather than ``"different event loop"`` — mirrors the
    connection-layer expired-weakref arm."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=1, timeout=1.0)

    loop_a = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(_bind_on_loop(pool))
    finally:
        loop_a.close()
    # Drop the strong ref so the weakref expires.
    import gc

    del loop_a
    gc.collect()

    loop_b = asyncio.new_event_loop()
    try:

        async def _drive() -> None:
            with pytest.raises(InterfaceError, match="closed event loop"):
                pool._check_loop_binding()

        loop_b.run_until_complete(_drive())
    finally:
        loop_b.close()
