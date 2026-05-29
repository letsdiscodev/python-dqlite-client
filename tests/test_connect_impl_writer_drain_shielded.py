"""``_connect_impl``'s finally writer drain wraps the bounded ``wait_for`` in
``asyncio.shield``: CancelledError (a BaseException) escapes ``suppress(Exception)``,
so an unshielded outer cancel orphaned the StreamReader task. The suppress is
narrowed to ``(OSError, TimeoutError)`` so non-transport raises surface."""

from __future__ import annotations

import inspect

from dqliteclient.connection import DqliteConnection


def _strip_comments_and_docstrings(src: str) -> str:
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


def test_connect_impl_finally_writer_drain_uses_shield() -> None:
    """The finally writer drain is wrapped in ``asyncio.shield`` so an outer
    cancel does not orphan the StreamReader task."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "asyncio.shield(inner_drain)" in code_only, (
        "_connect_impl's finally writer-drain must wrap the bounded "
        "wait_for in asyncio.shield so an outer cancel does not "
        "orphan the StreamReader task (mirrors cluster.py's "
        "_query_leader / open_admin_connection)."
    )


def test_connect_impl_finally_writer_drain_narrows_suppress() -> None:
    """The finally suppress is narrowed to ``(OSError, TimeoutError)`` so
    non-transport raises surface instead of being swallowed."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "contextlib.suppress(OSError, TimeoutError)" in code_only, (
        "_connect_impl's finally writer-drain must suppress only "
        "(OSError, TimeoutError); a broader Exception suppress "
        "silently absorbs AssertionError from mocks and other "
        "non-transport refactor bugs."
    )


def test_connect_impl_finally_writer_drain_attaches_observer() -> None:
    """The shielded inner drain attaches ``_observe_drain_exception`` so a
    post-cancel inner TimeoutError is observed and not logged at GC."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "add_done_callback(_observe_drain_exception)" in code_only, (
        "_connect_impl's shielded inner drain must attach "
        "_observe_drain_exception so a post-cancel TimeoutError on "
        "the inner wait_for is observed and asyncio does not log "
        "'Task exception was never retrieved' at GC."
    )
