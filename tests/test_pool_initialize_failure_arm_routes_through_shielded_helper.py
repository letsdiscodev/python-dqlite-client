"""Pin: ``ConnectionPool.initialize`` routes the post-gather
``if failures:`` survivor-close walk through the canonical
``_initialize_close_unqueued`` helper. An outer cancel landing
mid-walk does NOT orphan the remaining survivor transports.

The prior shape ran a bare ``for conn in successes: await
conn.close()`` loop with no per-iter shield, no cancel absorb,
and no observer on the implicit close task. An outer cancel
landing on the k-th survivor propagated immediately, leaking the
remaining ``len(successes) - k - 1`` live transports (open
sockets, ``weakref.finalize`` registration, reader Task,
server-side session) AND supplanted the ``failures[0]`` cause
with the cancel — operators saw the wrong root cause.

Symmetric with the already-hardened
``_initialize_close_unqueued`` helper (the put-loop tail
cleanup) which uses the same shielded-per-iter shape; the
``if failures:`` arm was the residual asymmetry.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient.pool import ConnectionPool


def _initialize_source() -> str:
    return textwrap.dedent(inspect.getsource(ConnectionPool.initialize))


def test_initialize_failure_arm_does_not_bare_await_close_in_successes_loop() -> None:
    """Structural pin: the ``initialize`` source must not contain
    a ``for conn in successes:`` loop that bare-awaits
    ``conn.close()``. The success-cleanup walk should route
    through ``_initialize_close_unqueued`` (or an equivalent
    shielded helper).
    """
    src = _initialize_source()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # Look for the bad shape: ``for <name> in successes: ... await
        # <name>.close()``.
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
    """The post-fix shape passes ``successes`` to the canonical
    shielded helper. Pin the routing so a future contributor
    cannot revert to a bare ``for conn in successes:`` loop.
    """
    src = _initialize_source()
    # Look for a call ``self._initialize_close_unqueued(successes)``.
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
