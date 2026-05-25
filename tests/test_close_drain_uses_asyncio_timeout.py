"""Pin: close-path drain sites use ``asyncio.timeout`` cancel-scope
semantics inside the inner Task, not ``asyncio.wait_for``.

The send/read sites in ``DqliteProtocol`` migrated from
``asyncio.wait_for`` to ``asyncio.timeout`` so an outer cancel landing
while the awaited coroutine has a return value would not discard that
value. The close-path drains (``DqliteConnection._connect_impl``,
``_close_impl``, ``_abort_protocol``, ``_invalidate._bounded_drain``,
``ClusterClient._query_leader``, ``ClusterClient.open_admin_connection``)
must use the same discipline so a future refactor that gives
``wait_closed()`` a return value does not silently regress.

Today ``writer.wait_closed()`` returns ``None`` so the regression would
be invisible at runtime — pin the source shape instead. The discipline
contract is: no ``asyncio.wait_for(... wait_closed() ...)`` calls in
the connection / cluster modules; use ``async with asyncio.timeout(...):
await ... wait_closed()`` instead.
"""

from __future__ import annotations

import inspect
import re

from dqliteclient import cluster, connection


def test_no_wait_for_around_wait_closed_in_connection_module() -> None:
    """Source-level pin: ``connection.py`` must not wrap
    ``wait_closed()`` in ``asyncio.wait_for(...)`` — the migration
    placed ``async with asyncio.timeout(...)`` inside an inner drain
    coroutine instead. Mirrors the discipline at
    ``protocol.py::_send`` and ``_read_data``.
    """
    src = inspect.getsource(connection)
    # Strip Python comments so cross-references in docstrings or
    # comments do not trip the pin.
    code = "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines() if line.strip())
    bad = re.findall(r"asyncio\.wait_for\([^)]*wait_closed", code)
    assert not bad, (
        "close-path drains in connection.py must use asyncio.timeout "
        f"cancel-scope shape, not asyncio.wait_for; found: {bad!r}"
    )


def test_no_wait_for_around_wait_closed_in_cluster_module() -> None:
    """Source-level pin: ``cluster.py`` must not wrap ``wait_closed()``
    in ``asyncio.wait_for(...)``. ``_query_leader`` and
    ``open_admin_connection`` migrated to the inner ``asyncio.timeout``
    shape.
    """
    src = inspect.getsource(cluster)
    code = "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines() if line.strip())
    bad = re.findall(r"asyncio\.wait_for\([^)]*wait_closed", code)
    assert not bad, (
        "close-path drains in cluster.py must use asyncio.timeout "
        f"cancel-scope shape, not asyncio.wait_for; found: {bad!r}"
    )


def test_connection_module_uses_asyncio_timeout_around_wait_closed() -> None:
    """Positive pin: at least one drain site in ``connection.py`` uses
    the migrated ``async with asyncio.timeout(...): await ...
    wait_closed()`` shape. Guards against accidental removal of the
    discipline at all sites.
    """
    src = inspect.getsource(connection)
    # Look for the inner-task drain coroutine shape.
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
    """Positive pin for ``cluster.py``: ``_query_leader`` and
    ``open_admin_connection`` both use the migrated inner-timeout shape.
    """
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
