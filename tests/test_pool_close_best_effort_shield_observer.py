"""Pin: ``ConnectionPool._close_best_effort`` schedules the inner
``conn.close()`` as an explicit ``Task`` with
``_observe_drain_exception`` attached, then awaits
``asyncio.shield(task)``.

The prior shape — ``await asyncio.shield(conn.close())`` — wraps
the bare coroutine in an IMPLICIT task. If the outer awaiter is
cancelled or otherwise abandoned while the inner close is still
running, the implicit task is left running unobserved. When the
close eventually raises, asyncio's task finaliser surfaces the
exception via ``Task was destroyed but it is pending`` (if the
loop closes mid-flight) or "exception was never retrieved" at GC.

The canonical fix shape (already established at
``pool.py``'s ``_drain_idle`` per-iteration site and at
``connection.py``'s ``_abort_protocol``) is::

    inner = asyncio.ensure_future(conn.close())
    inner.add_done_callback(_observe_drain_exception)
    await asyncio.shield(inner)

This pin is structural via source inspection because the runtime
behaviour (asyncio's ``_log_on_exception`` for shielded futures)
is intentional and orthogonal to the explicit-task discipline the
project relies on for cross-site uniformity. Regression risk is
that a future contributor inlines the bare-coro shield pattern;
the AST walk catches that mechanically.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient import pool as pool_mod


def _close_best_effort_source() -> str:
    return textwrap.dedent(inspect.getsource(pool_mod.ConnectionPool._close_best_effort))


def test_close_best_effort_uses_explicit_task_and_done_callback() -> None:
    """The helper body must construct an explicit Task for the inner
    ``conn.close()``, attach a done-callback, and shield the Task
    (not the bare coroutine).
    """
    src = _close_best_effort_source()
    tree = ast.parse(src)

    seen_ensure_future = False
    seen_add_done_callback = False
    seen_shield_of_named_task = False
    seen_bare_shield_of_close = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # ``asyncio.ensure_future(conn.close())``
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "ensure_future"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and node.args[0].func.attr == "close"
        ):
            seen_ensure_future = True
        # ``<something>.add_done_callback(...)``
        if isinstance(func, ast.Attribute) and func.attr == "add_done_callback":
            seen_add_done_callback = True
        # ``asyncio.shield(<Name>)`` (named Task, not a bare coroutine call)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "shield"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
        ):
            if node.args and isinstance(node.args[0], ast.Name):
                seen_shield_of_named_task = True
            # Regression detector: ``asyncio.shield(conn.close())``
            if node.args and isinstance(node.args[0], ast.Call):
                inner = node.args[0]
                if isinstance(inner.func, ast.Attribute) and inner.func.attr == "close":
                    seen_bare_shield_of_close = True

    assert seen_ensure_future, (
        "_close_best_effort must wrap conn.close() in an explicit Task via "
        "asyncio.ensure_future so the close-driven exception state lives on "
        "a tracked object (mirrors the _drain_idle per-iteration site at "
        "pool.py:1493)"
    )
    assert seen_add_done_callback, (
        "_close_best_effort must attach a done-callback (canonically "
        "_observe_drain_exception) to the explicit Task so the abandoned "
        "close's eventual exception is retrieved silently"
    )
    assert seen_shield_of_named_task, (
        "_close_best_effort must shield the explicit Task, not a bare "
        "coroutine — asyncio.shield(named_task) instead of "
        "asyncio.shield(conn.close())"
    )
    assert not seen_bare_shield_of_close, (
        "_close_best_effort must NOT call asyncio.shield(conn.close()) "
        "directly — the bare-coro pattern orphans the implicit Task on "
        "outer-await abandonment. Use ensure_future + add_done_callback + "
        "shield(task) instead."
    )
