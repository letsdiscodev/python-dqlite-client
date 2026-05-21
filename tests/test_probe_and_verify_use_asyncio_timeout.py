"""Pin: ``_probe_one`` (parallel-sweep arm) and ``_verify_redirect``
use ``async with asyncio.timeout(...)`` (cancel-scope semantics)
rather than ``asyncio.wait_for`` (which discards the inner task's
pending result on outer-cancel — the canonical leak shape).

Sibling fast-path discipline pinned at
``test_cached_fast_path_uses_asyncio_timeout``; this fence keeps
the two remaining probe call sites in the same module aligned with
the project-wide cancel-scope discipline.
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
            if stripped.count('"""') + stripped.count("'''") >= 2:
                continue
            in_docstring = True
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_probe_one_uses_asyncio_timeout_not_wait_for() -> None:
    """The parallel-sweep probe block in ``_find_leader_impl``'s
    nested ``_probe_one`` must call the inner ``_query_leader`` under
    ``async with asyncio.timeout(...)``, not ``asyncio.wait_for``."""
    src = inspect.getsource(ClusterClient._find_leader_impl)
    # Cut to the ``_probe_one`` body: from its def to the end of
    # ``_find_leader_impl``.
    start = src.find("async def _probe_one")
    assert start != -1, "could not locate _probe_one nested def"
    probe_block = src[start:]
    code_only = _strip_comments_and_docstrings(probe_block)
    assert "asyncio.wait_for(" not in code_only, (
        "_probe_one must use ``async with asyncio.timeout(...)`` "
        "around the inner ``_query_leader`` call (cancel-scope "
        "semantics). ``asyncio.wait_for`` discards the inner task's "
        "pending result on outer-cancel — the canonical leak shape."
    )
    assert "async with asyncio.timeout(self._attempt_timeout)" in code_only


def test_verify_redirect_uses_asyncio_timeout_not_wait_for() -> None:
    """``_verify_redirect`` must use ``async with asyncio.timeout(...)``
    so an outer cancel landing while ``_query_leader`` is just
    returning the verified address does NOT discard it — preserves
    the cache-population path the fast-path arm relies on."""
    src = inspect.getsource(ClusterClient._verify_redirect)
    code_only = _strip_comments_and_docstrings(src)
    assert "asyncio.wait_for(" not in code_only, (
        "_verify_redirect must use ``async with asyncio.timeout(...)`` "
        "around the inner ``_query_leader`` call so a race with outer "
        "cancel does not discard the verified-address result."
    )
    assert "async with asyncio.timeout(effective_timeout)" in code_only
