"""Close-path drain sites use ``asyncio.timeout`` cancel-scope semantics, not
``asyncio.wait_for``, so an outer cancel cannot discard a return value.

``wait_closed()`` returns None today, so a regression is runtime-invisible —
pin the source shape instead.
"""

from __future__ import annotations

import inspect
import re

from dqliteclient import cluster, connection


def test_no_wait_for_around_wait_closed_in_connection_module() -> None:
    """``connection.py`` must not wrap ``wait_closed()`` in ``asyncio.wait_for``."""
    src = inspect.getsource(connection)
    # Strip comments so cross-references in them do not trip the pin.
    code = "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines() if line.strip())
    bad = re.findall(r"asyncio\.wait_for\([^)]*wait_closed", code)
    assert not bad, (
        "close-path drains in connection.py must use asyncio.timeout "
        f"cancel-scope shape, not asyncio.wait_for; found: {bad!r}"
    )


def test_no_wait_for_around_wait_closed_in_cluster_module() -> None:
    """``cluster.py`` must not wrap ``wait_closed()`` in ``asyncio.wait_for``."""
    src = inspect.getsource(cluster)
    code = "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines() if line.strip())
    bad = re.findall(r"asyncio\.wait_for\([^)]*wait_closed", code)
    assert not bad, (
        "close-path drains in cluster.py must use asyncio.timeout "
        f"cancel-scope shape, not asyncio.wait_for; found: {bad!r}"
    )


def test_connection_module_uses_asyncio_timeout_around_wait_closed() -> None:
    """At least one drain site in ``connection.py`` uses the
    ``async with asyncio.timeout(...): await ...wait_closed()`` shape."""
    src = inspect.getsource(connection)
    pattern = re.compile(
        r"async with asyncio\.timeout\([^)]+\):\s*\n\s*await [^.]+\.wait_closed",
        re.MULTILINE,
    )
    assert pattern.search(src), (
        "connection.py must contain at least one "
        "`async with asyncio.timeout(...): await ...wait_closed()` "
        "drain shape (migration from asyncio.wait_for)"
    )


def test_cluster_module_uses_asyncio_timeout_around_wait_closed() -> None:
    """``_query_leader`` and ``open_admin_connection`` both use the inner-timeout shape."""
    src = inspect.getsource(cluster)
    pattern = re.compile(
        r"async with asyncio\.timeout\([^)]+\):\s*\n\s*await [^.]+\.wait_closed",
        re.MULTILINE,
    )
    matches = pattern.findall(src)
    assert len(matches) >= 2, (
        "cluster.py must contain at least two "
        "`async with asyncio.timeout(...): await ...wait_closed()` "
        "drain shapes (one each for _query_leader and "
        f"open_admin_connection); found {len(matches)}"
    )
