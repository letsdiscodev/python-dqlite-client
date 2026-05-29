"""_probe_one wraps the semaphore acquire+store in a try/except
(KeyboardInterrupt, SystemExit) that releases the permit before re-raising: a
signal landing in the bytecode window between acquire and `sem_acquired = True`
would otherwise leak the permit and wedge the find_leader sweep.

Structural pin via source inspection: pytest-asyncio's runner intercepts both
signals before an in-test capture could observe a runtime version.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient import cluster as cluster_mod


def _find_probe_one_source() -> str:
    """Return the source of the _probe_one closure inside _find_leader_impl."""
    enclosing_src = textwrap.dedent(inspect.getsource(cluster_mod.ClusterClient._find_leader_impl))
    module = ast.parse(enclosing_src)
    for inner in ast.walk(module):
        if isinstance(inner, ast.AsyncFunctionDef) and inner.name == "_probe_one":
            return ast.unparse(inner)
    raise AssertionError(
        "could not locate _probe_one inside ClusterClient._find_leader_impl source"
    )


def test_probe_one_wraps_semaphore_acquire_store_in_signal_safe_arm() -> None:
    """The acquire+store pair lives in a try whose (KeyboardInterrupt, SystemExit)
    arm releases the semaphore before re-raising."""
    src = _find_probe_one_source()

    assert "semaphore.acquire()" in src, (
        "_probe_one source no longer contains 'semaphore.acquire()'; "
        "test needs updating to match the new shape"
    )
    assert "except (KeyboardInterrupt, SystemExit)" in src, (
        "_probe_one must wrap the semaphore acquire/store pair in a "
        "try/except (KeyboardInterrupt, SystemExit) arm so a signal "
        "landing post-acquire releases the permit (mirror of the "
        "threading-lock discipline at dqlitedbapi.connection's "
        "_loop_lock acquire arm)"
    )
    # The except arm must contain a release call, not just the clause text.
    tree = ast.parse(src)
    found_signal_safe_release = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        if isinstance(node.type, ast.Tuple):
            names = {elt.id for elt in node.type.elts if isinstance(elt, ast.Name)}
            if {"KeyboardInterrupt", "SystemExit"} <= names:
                for body_node in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if (
                        isinstance(body_node, ast.Call)
                        and isinstance(body_node.func, ast.Attribute)
                        and body_node.func.attr == "release"
                        and isinstance(body_node.func.value, ast.Name)
                        and body_node.func.value.id == "semaphore"
                    ):
                        found_signal_safe_release = True
                        break
    assert found_signal_safe_release, (
        "the except (KeyboardInterrupt, SystemExit) arm must call "
        "semaphore.release() so the permit is restored before the "
        "BaseException propagates"
    )


def test_probe_one_outer_finally_release_safety_net_remains() -> None:
    """The outer finally `if sem_acquired: semaphore.release()` safety net remains:
    every non-signal exception path still relies on it for permit release."""
    src = _find_probe_one_source()
    assert "if sem_acquired:" in src, "outer finally safety-net release was removed"
    assert "semaphore.release()" in src, "no semaphore.release() call remains"
