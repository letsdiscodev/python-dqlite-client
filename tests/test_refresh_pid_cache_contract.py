"""Pin: ``_refresh_pid_cache`` mutates the module-level
``_current_pid`` and is registered with ``os.register_at_fork``.

This is the producer side of the cycle-21 pid-cache fork-detection
contract. The fork-guard tests exercise the *consumer* side
(``_current_pid != _creator_pid`` raising ``InterfaceError``); they
sidestep the cache by mutating ``_creator_pid`` directly. Coverage
of the actual mutation body inside ``_refresh_pid_cache`` is
structurally invisible to a unit-test run because it executes in
the forked child, which calls ``os._exit(0)`` before flushing
coverage data.

A regression that drops ``global _current_pid`` from
``_refresh_pid_cache`` (Python's "forgotten global" footgun) would
make the assignment local-scoped, leave the module attribute at
the parent's pid forever, and the entire fork-after-init
detection mechanism would silently disable. Without a unit test on
the producer side, the regression would clear unit CI invisibly.

Pin three properties:
1. The function name exists and is callable.
2. The function actually mutates the module attribute (call it,
   then check the result by patching ``os.getpid`` to a sentinel).
3. The function is registered as an ``after_in_child`` fork hook.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from dqliteclient import connection as conn_mod


def test_refresh_pid_cache_mutates_module_attribute() -> None:
    """Calling _refresh_pid_cache must update connection._current_pid."""
    saved = conn_mod._current_pid
    sentinel = saved + 17
    try:
        with patch("dqliteclient.connection.os.getpid", return_value=sentinel):
            conn_mod._refresh_pid_cache()
        assert conn_mod._current_pid == sentinel, (
            "_refresh_pid_cache must assign os.getpid() to the module-level "
            "_current_pid; if a refactor dropped 'global _current_pid', the "
            "assignment becomes local-scoped and the fork-detection guard "
            "silently disables in forked children."
        )
    finally:
        # Restore so subsequent tests in the same process see the
        # real pid (the registered after_in_child callback also
        # restores it on the next fork, but other tests run in the
        # same process and rely on the cache being correct).
        conn_mod._current_pid = saved


def test_refresh_pid_cache_is_callable_zero_arg() -> None:
    """``os.register_at_fork(after_in_child=...)`` requires a
    zero-arg callable. Pin the signature directly so a refactor
    that adds a parameter (silently breaking the fork hook
    registration's call shape) is caught."""
    saved = conn_mod._current_pid
    try:
        result = conn_mod._refresh_pid_cache()
    finally:
        conn_mod._current_pid = saved
    assert result is None  # PEP 257-style: side-effect function returns None


def test_after_in_child_fork_actually_refreshes_cache() -> None:
    """End-to-end: do an actual fork and confirm the child's
    ``_current_pid`` matches the child's ``os.getpid()``. This is
    the truest test of the producer-side contract — coverage
    tooling can't see the child's execution but a pipe-based child→
    parent assertion-result reporter can."""
    if not hasattr(os, "fork"):
        return
    parent_pid = os.getpid()
    assert conn_mod._current_pid == parent_pid

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            try:
                # In the child, after_in_child has fired.
                # _current_pid should now equal the child's pid.
                child_pid = os.getpid()
                cached = conn_mod._current_pid
                if cached == child_pid and cached != parent_pid:
                    os.write(w, b"OK")
                else:
                    os.write(
                        w,
                        f"FAIL: child_pid={child_pid} cached={cached} "
                        f"parent_pid={parent_pid}".encode(),
                    )
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
