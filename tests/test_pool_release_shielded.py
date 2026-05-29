"""Pin: acquire()'s exception path, _release, and _drain_idle's finally all wrap
``_release_reservation()`` in ``asyncio.shield``. Without it an outer timeout firing
mid-decrement leaks _size, drifting the pool toward max_size with no open
connections until deadlock. Source-inspected because the race timing is unpinnable.
"""

from __future__ import annotations

import inspect

import dqliteclient.pool


def _source_of(method_name: str) -> str:
    cls = dqliteclient.pool.ConnectionPool
    return inspect.getsource(getattr(cls, method_name))


class TestReleaseReservationShielded:
    def test_release_has_shielded_release_reservation(self) -> None:
        """_release shields _release_reservation so cancellation cannot interrupt
        the size-counter decrement."""
        source = _source_of("_release")
        assert "asyncio.shield" in source or "_safe_release" in source, (
            "_release must shield _release_reservation against cancellation"
        )
        bare_calls = source.count("await self._release_reservation()")
        shielded_calls = source.count("asyncio.shield(self._release_reservation())")
        assert shielded_calls >= bare_calls, (
            f"found bare _release_reservation() call without shield in _release: "
            f"bare={bare_calls} shielded={shielded_calls}"
        )

    def test_drain_idle_has_shielded_release_reservation(self) -> None:
        """_drain_idle's finally shields _release_reservation; otherwise an outer
        cancel on the helper's lock acquire leaves _size inconsistent."""
        source = _source_of("_drain_idle")
        assert "asyncio.shield" in source, "_drain_idle must shield _release_reservation"
