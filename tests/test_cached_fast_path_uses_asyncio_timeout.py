"""Pin: ``_find_leader_impl``'s cached-leader fast path uses
``async with asyncio.timeout(...)`` rather than
``await asyncio.wait_for(...)``.

Three sibling sites in the same module document the discipline —
``asyncio.timeout`` (cancel-scope semantics) lets the inner
coroutine's ``finally:`` drain transport state cleanly under outer
cancel, while ``asyncio.wait_for`` cancels the inner task and
discards its pending result (the canonical leak shape: orphaned
writer transports / drain tasks). The cached fast path was the one
asymmetric outlier; this pin keeps it in line.

Sibling sites: ``_query_leader`` outer-dial scope,
``open_admin_connection`` envelope, and ``_connect_impl`` envelope.
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop ``#`` comment lines and contiguous triple-quoted strings
    so a substring fence on the actual call sites cannot be tripped
    by explanatory comments mentioning the rejected primitive."""
    lines = []
    in_docstring = False
    for line in src.splitlines():
        stripped = line.strip()
        if in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Single-line docstring vs opener; if it closes on the
            # same line (count of triple-quotes >= 2), skip just this
            # line; otherwise enter docstring mode.
            if stripped.count('"""') + stripped.count("'''") >= 2:
                continue
            in_docstring = True
            continue
        # Drop full-line ``#`` comments; keep code with trailing
        # comments unchanged (the rejected primitive is unlikely to
        # appear in a trailing comment, and we want a tight fence).
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_cached_fast_path_block_does_not_call_asyncio_wait_for() -> None:
    """Substring fence on call sites: the cached fast-path block in
    ``_find_leader_impl`` must NOT call ``asyncio.wait_for`` directly.
    If a future refactor reintroduces the rejected primitive on the
    cached fast path, this test trips so the reviewer sees the
    established discipline immediately.

    Scoped to the cached fast-path block (the lines between
    ``cached = self._get_last_known_leader()`` and the cache-miss
    fall-through ``cached``-arm exit) so the parallel-sweep arm's
    own ``asyncio.wait_for`` (a different design choice) is not
    swept up by the fence.

    Comments / docstrings are stripped so explanatory references to
    ``asyncio.wait_for`` (the WHY this is rejected) do not trip the
    fence."""
    src = inspect.getsource(ClusterClient._find_leader_impl)
    # Cut the fast-path block: from the ``cached = `` line until the
    # parallel-sweep header (``await asyncio.gather`` is too late;
    # use the ``_probe_one`` nested-def site as the boundary).
    start = src.find("cached = self._get_last_known_leader()")
    end = src.find("async def _probe_one", start)
    assert start != -1, "could not locate cached fast-path block start"
    assert end != -1, "could not locate cached fast-path block end"
    fast_path_block = src[start:end]
    code_only = _strip_comments_and_docstrings(fast_path_block)
    assert "asyncio.wait_for(" not in code_only, (
        "The cached fast path in _find_leader_impl must use "
        "``async with asyncio.timeout(...)`` (cancel-scope semantics), "
        "mirroring the discipline at _query_leader / "
        "open_admin_connection / _connect_impl. ``asyncio.wait_for`` "
        "discards the inner task's pending result on outer-cancel "
        "and leaves the writer/drain orphaned."
    )


def test_find_leader_impl_uses_asyncio_timeout_for_cached_probe() -> None:
    """Mirror assertion: the cached fast path uses
    ``async with asyncio.timeout(...)`` to scope the cached probe."""
    src = inspect.getsource(ClusterClient._find_leader_impl)
    assert "async with asyncio.timeout(self._attempt_timeout)" in src
