"""Pool reservation release must survive outer cancellation.

The exception-path of ``acquire()`` wraps ``_release_reservation()``
in ``asyncio.shield`` (ISSUE-76 / ISSUE-198 / ISSUE-300). The
symmetric **success-path** ``_release`` and the ``_drain_idle``
``finally`` block did not. An outer ``asyncio.timeout`` that fires
during a lock-contended reservation decrement left ``_size``
incremented permanently — repeatable occurrences drift the pool
toward ``max_size`` with no actual open connections and eventually
deadlock.

These are regression fences pinning the shield pattern at both call
sites. They inspect the source to confirm the shield is in place
alongside the release calls — behavioral tests of the race are
inherently flaky because the cancel timing can never be guaranteed
to land inside the specific microsecond window.
"""

from __future__ import annotations

import inspect

import dqliteclient.pool


def _source_of(method_name: str) -> str:
    cls = dqliteclient.pool.ConnectionPool
    return inspect.getsource(getattr(cls, method_name))


class TestReleaseReservationShielded:
    def test_release_has_shielded_release_reservation(self) -> None:
        """The success-path of _release MUST wrap _release_reservation
        in asyncio.shield so outer cancellation cannot interrupt the
        size-counter decrement.
        """
        source = _source_of("_release")
        # Three release paths: Pool closed, reset failed, QueueFull.
        # Each must go through a shielded release.
        assert "asyncio.shield" in source or "_safe_release" in source, (
            "_release must shield _release_reservation against cancellation; "
            "see ISSUE-312 + exception-path precedent at pool.py:546-547"
        )
        # The bare `await self._release_reservation()` calls must be gone.
        # At most they are inside a shield/suppress wrapper.
        bare_calls = source.count("await self._release_reservation()")
        shielded_calls = source.count("asyncio.shield(self._release_reservation())")
        assert shielded_calls >= bare_calls, (
            f"found bare _release_reservation() call without shield in _release: "
            f"bare={bare_calls} shielded={shielded_calls}"
        )

    def test_drain_idle_has_shielded_release_reservation(self) -> None:
        """The finally block in _drain_idle MUST shield
        _release_reservation — otherwise an outer cancel landing on the
        lock acquire inside the helper leaves _size inconsistent.
        (ISSUE-139 deferred this; ISSUE-314 files it.)
        """
        source = _source_of("_drain_idle")
        assert "asyncio.shield" in source, (
            "_drain_idle must shield _release_reservation; see ISSUE-314"
        )
