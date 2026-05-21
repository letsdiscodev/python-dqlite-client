"""Pin: ``DqliteConnection._connect_impl``'s finally-clause writer
drain is shielded against outer cancel.

Prior shape:

    if writer is not None:
        writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer.wait_closed(), timeout=...)

``asyncio.CancelledError`` is a ``BaseException`` subclass and
propagated past ``suppress(Exception)``. An outer cancel landing
inside the finally orphaned the StreamReader task spawned by
``open_connection``; the next GC sweep printed
"Task was destroyed but it is pending".

The fix wraps the bounded ``wait_for`` in ``asyncio.shield`` (so
the inner drain runs to completion even when the outer awaiter is
cancelled) and narrows the suppress to ``(OSError, TimeoutError)``
so refactor-time non-transport raises surface.
"""

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
    """The finally arm's writer drain must be wrapped in
    ``asyncio.shield`` so an outer cancel does not orphan the
    StreamReader task. Mirrors ``cluster.py``'s ``_query_leader`` /
    ``open_admin_connection`` finally arms.
    """
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "asyncio.shield(inner_drain)" in code_only, (
        "_connect_impl's finally writer-drain must wrap the bounded "
        "wait_for in asyncio.shield so an outer cancel does not "
        "orphan the StreamReader task (mirrors cluster.py's "
        "_query_leader / open_admin_connection)."
    )


def test_connect_impl_finally_writer_drain_narrows_suppress() -> None:
    """The finally arm's suppress must be narrowed from
    ``Exception`` to ``(OSError, TimeoutError)`` so unexpected
    non-transport raises (mock ``AssertionError``, custom
    ``dial_func`` raising) surface instead of being silently
    swallowed.
    """
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    # The finally drain block ends with the shield await. Look for
    # the narrowed suppress immediately preceding it.
    assert "contextlib.suppress(OSError, TimeoutError)" in code_only, (
        "_connect_impl's finally writer-drain must suppress only "
        "(OSError, TimeoutError); a broader Exception suppress "
        "silently absorbs AssertionError from mocks and other "
        "non-transport refactor bugs."
    )


def test_connect_impl_finally_writer_drain_attaches_observer() -> None:
    """The shielded inner drain task must have
    ``_observe_drain_exception`` attached so a TimeoutError on the
    inner ``wait_for`` after the outer cancel does not surface as
    asyncio's "Task exception was never retrieved" GC warning.
    """
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "add_done_callback(_observe_drain_exception)" in code_only, (
        "_connect_impl's shielded inner drain must attach "
        "_observe_drain_exception so a post-cancel TimeoutError on "
        "the inner wait_for is observed and asyncio does not log "
        "'Task exception was never retrieved' at GC."
    )
