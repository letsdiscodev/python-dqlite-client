"""ConnectionPool must have an informative __repr__.

ISSUE-14 added reprs to Connection / Cursor / DqliteConnection; the
pool class was overlooked and fell back to
``<ConnectionPool object at 0x...>``. The repr surfaces
``size`` / ``min_size`` / ``max_size`` / ``open|closed`` and a
capped address list for debug logs and interactive exploration;
no secrets (dqlite has no wire auth) are revealed.
"""

from __future__ import annotations

from dqliteclient.pool import ConnectionPool


def test_repr_shows_size_and_state() -> None:
    pool = ConnectionPool(["h1:9001", "h2:9002"], min_size=0, max_size=4)
    r = repr(pool)
    assert r.startswith("ConnectionPool(")
    assert "h1:9001" in r
    assert "size=0" in r
    assert "min_size=0" in r
    assert "max_size=4" in r
    assert "open" in r


def test_repr_shows_closed_state() -> None:
    pool = ConnectionPool(["h1:9001"], min_size=0, max_size=2)
    pool._closed = True
    assert "closed" in repr(pool)


def test_repr_caps_long_address_list() -> None:
    addrs = [f"h{i}:9001" for i in range(10)]
    pool = ConnectionPool(addrs, min_size=0, max_size=2)
    r = repr(pool)
    assert "+7" in r  # 10 total, first 3 shown, +7 hint
    assert "h3" not in r  # 4th not shown directly
