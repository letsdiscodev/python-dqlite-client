"""Cached-leader fast path uses ``asyncio.timeout``, not ``asyncio.wait_for``.

``asyncio.timeout`` cancel-scope semantics let the inner coroutine's ``finally:``
drain cleanly under outer cancel; ``wait_for`` discards the inner result, orphaning
writer transports / drain tasks.
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop comment lines and triple-quoted strings so explanatory mentions of
    the rejected primitive cannot trip the substring fence on call sites."""
    lines = []
    in_docstring = False
    for line in src.splitlines():
        stripped = line.strip()
        if in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Single-line docstring closes on the same line (>= 2 triple-quotes).
            if stripped.count('"""') + stripped.count("'''") >= 2:
                continue
            in_docstring = True
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_cached_fast_path_block_does_not_call_asyncio_wait_for() -> None:
    """The cached fast-path block must not call ``asyncio.wait_for`` directly.

    Scoped to the fast-path block so the parallel-sweep arm's own ``wait_for``
    (a deliberate choice) is not swept up.
    """
    src = inspect.getsource(ClusterClient._find_leader_impl)
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
    """The cached fast path uses ``async with asyncio.timeout(...)``."""
    src = inspect.getsource(ClusterClient._find_leader_impl)
    assert "async with asyncio.timeout(self._attempt_timeout)" in src
