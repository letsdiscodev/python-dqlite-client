"""Pin: ``ConnectionPool.initialize`` routes the failure-arm survivor-close
walk through ``_initialize_close_unqueued`` so a mid-walk cancel leaks none."""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient.pool import ConnectionPool


def _initialize_source() -> str:
    return textwrap.dedent(inspect.getsource(ConnectionPool.initialize))


def test_initialize_failure_arm_does_not_bare_await_close_in_successes_loop() -> None:
    """The ``initialize`` source must not bare-await ``conn.close()`` in a
    ``for conn in successes:`` loop."""
    src = _initialize_source()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # Bad shape: ``for <name> in successes: ... await <name>.close()``.
        iter_expr = node.iter
        if not (isinstance(iter_expr, ast.Name) and iter_expr.id == "successes"):
            continue
        target = node.target
        if not isinstance(target, ast.Name):
            continue
        loop_var = target.id
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Await):
                continue
            call = inner.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "close"
                and isinstance(func.value, ast.Name)
                and func.value.id == loop_var
            ):
                raise AssertionError(
                    "initialize's ``for conn in successes:`` loop still "
                    "bare-awaits ``conn.close()`` — an outer cancel "
                    "mid-walk leaks the remaining survivors. Route "
                    "through ``_initialize_close_unqueued`` (or a "
                    "shielded equivalent) so each close is per-iter "
                    "shielded and CancelledError is absorbed without "
                    "breaking the walk."
                )


def test_initialize_failure_arm_calls_initialize_close_unqueued_with_successes() -> None:
    """The failure arm passes ``successes`` to ``_initialize_close_unqueued``."""
    src = _initialize_source()
    tree = ast.parse(src)
    found_helper_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "_initialize_close_unqueued"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "successes"
        ):
            found_helper_call = True
    assert found_helper_call, (
        "initialize's failure arm should call "
        "``self._initialize_close_unqueued(successes)`` to route the "
        "success-cleanup walk through the shielded helper. The helper "
        "absorbs CancelledError per-iter so ``raise failures[0]`` "
        "below carries the actual initialise failure cause instead of "
        "the late cancel."
    )
