"""Pin: ``_probe_one`` wraps the semaphore acquire+store pair in a
``try/except (KeyboardInterrupt, SystemExit)`` arm that releases the
permit before re-raising.

The prior shape — bare ``await semaphore.acquire()`` followed by
``sem_acquired = True`` — has a 1-3 bytecode signal-delivery window
between the acquire returning (permit decremented) and the
bookkeeping store. A ``KeyboardInterrupt`` / ``SystemExit`` /
``PyErr_SetAsyncExc`` landing there leaks the permit permanently:
the outer ``finally``'s ``if sem_acquired: semaphore.release()``
guard misfires because ``sem_acquired`` is still ``False``. With
``_concurrent_leader_conns=10`` (default) a handful of these wedges
the entire ``find_leader`` sweep.

Mirrors the threading-lock discipline at ``dqlitedbapi.connection``'s
``_loop_lock`` acquire arm.

This is a structural pin via source inspection: a runtime-level
test is impractical because pytest-asyncio's runner intercepts
both ``KeyboardInterrupt`` and ``SystemExit`` (the two BaseException
subclasses the production ``except`` arm targets) as abort signals
before any in-test capture can observe them.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient import cluster as cluster_mod


def _find_probe_one_source() -> str:
    """Locate the ``_probe_one`` nested closure inside
    ``ClusterClient._find_leader_impl`` and return its source code."""
    enclosing_src = textwrap.dedent(inspect.getsource(cluster_mod.ClusterClient._find_leader_impl))
    module = ast.parse(enclosing_src)
    for inner in ast.walk(module):
        if isinstance(inner, ast.AsyncFunctionDef) and inner.name == "_probe_one":
            return ast.unparse(inner)
    raise AssertionError(
        "could not locate _probe_one inside ClusterClient._find_leader_impl source"
    )


def test_probe_one_wraps_semaphore_acquire_store_in_signal_safe_arm() -> None:
    """The acquire+store pair must live inside a ``try`` whose
    ``except (KeyboardInterrupt, SystemExit)`` arm releases the
    semaphore before re-raising. Pin the structural shape so a
    future contributor cannot accidentally revert the discipline
    by inlining ``await semaphore.acquire(); sem_acquired = True``
    as bare statements outside the guard.
    """
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
    # Verify the except arm contains a release call so the structural
    # shape is enforced beyond the bare ``except`` clause text.
    tree = ast.parse(src)
    found_signal_safe_release = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        # Expect ``except (KeyboardInterrupt, SystemExit)``.
        if isinstance(node.type, ast.Tuple):
            names = {elt.id for elt in node.type.elts if isinstance(elt, ast.Name)}
            if {"KeyboardInterrupt", "SystemExit"} <= names:
                # Walk the arm body for a semaphore.release() call.
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
    """Regression guard: the outer ``finally: if sem_acquired:
    semaphore.release()`` safety net must still exist — the
    signal-safe arm covers the narrow acquire/store window, but
    every other exception path (DqliteConnectionError, ProtocolError,
    etc.) still relies on the finally for permit release.
    """
    src = _find_probe_one_source()
    assert "if sem_acquired:" in src, "outer finally safety-net release was removed"
    assert "semaphore.release()" in src, "no semaphore.release() call remains"
