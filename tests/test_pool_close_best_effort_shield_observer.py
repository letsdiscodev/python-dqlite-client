"""_close_best_effort must shield an explicit ensure_future(conn.close()) Task
with _observe_drain_exception attached, not the bare coroutine: shield(conn.close())
orphans the implicit task on outer-await abandonment, surfacing the eventual
exception at GC. Pinned structurally via AST walk."""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient import pool as pool_mod


def _close_best_effort_source() -> str:
    return textwrap.dedent(inspect.getsource(pool_mod.ConnectionPool._close_best_effort))


def test_close_best_effort_uses_explicit_task_and_done_callback() -> None:
    """Body must build an explicit Task for conn.close(), attach a done-callback,
    and shield the Task, not the bare coroutine."""
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
        # asyncio.ensure_future(conn.close())
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
