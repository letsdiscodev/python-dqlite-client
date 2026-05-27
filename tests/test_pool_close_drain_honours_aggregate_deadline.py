"""Pin: ``ConnectionPool.close()`` derives an aggregate close-budget
deadline and threads it through both ``_drain_idle`` and
``_drain_remaining_after_cancel`` so the SIGTERM contract bounds
end-to-end wall-clock regardless of stuck-close stampedes.

The prior shape called ``await self._drain_idle()`` (no
``deadline=`` kwarg). With the wire-default ``close_timeout=0.5``
and ``max_size=10``, a queue of pathologically-slow ``close()``
peers consumed ~20 s before the close path returned; with
``max_size=100``, ~200 s. Sibling acquire-path call sites
already passed ``deadline=...``; the close path was the
structural outlier.

``_drain_remaining_after_cancel`` (the cancel-recovery sweep)
gains the same ``deadline`` kwarg so an outer cancel mid-drain
cannot re-amplify the gap via the recovery arm.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient.pool import ConnectionPool


def _close_source() -> str:
    return textwrap.dedent(inspect.getsource(ConnectionPool.close))


def _drain_remaining_after_cancel_signature() -> inspect.Signature:
    return inspect.signature(ConnectionPool._drain_remaining_after_cancel)


def test_pool_close_passes_deadline_to_drain_idle() -> None:
    """Structural pin: ``pool.close()`` must call
    ``self._drain_idle(deadline=...)`` (not the bare
    ``self._drain_idle()`` shape that omits the aggregate
    close-budget gate).
    """
    src = _close_source()
    tree = ast.parse(src)
    found_with_deadline = False
    found_bare = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "_drain_idle"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            if any(kw.arg == "deadline" for kw in node.keywords):
                found_with_deadline = True
            else:
                found_bare = True
    assert found_with_deadline, (
        "pool.close() must pass ``deadline=<aggregate_close_budget>`` "
        "to ``self._drain_idle``. The aggregate-budget gate is what "
        "bounds end-to-end close wall-clock under a stuck-close "
        "stampede; without it, ``close()`` waits ``N × per_iter_cap`` "
        "even when ``close_timeout`` was sized per-peer."
    )
    assert not found_bare, (
        "pool.close() should NOT contain a bare ``self._drain_idle()`` "
        "call. Every drain invocation from the close path must thread "
        "the aggregate deadline through."
    )


def test_drain_remaining_after_cancel_accepts_deadline_kwarg() -> None:
    """The sibling cancel-recovery sweep must accept the same
    ``deadline`` plumbing so the close-budget gate is honoured
    end-to-end (outer cancel can't re-amplify via the recovery arm).
    """
    sig = _drain_remaining_after_cancel_signature()
    assert "deadline" in sig.parameters, (
        "_drain_remaining_after_cancel must accept a ``deadline`` "
        "kwarg so the cancel-recovery sweep honours the same "
        "aggregate close-budget the main drain loop enforces. "
        "Without it, an outer cancel re-amplifies the gap the "
        "main-loop deadline was meant to bound."
    )


def test_pool_close_forwards_deadline_into_drain_remaining_after_cancel() -> None:
    """Pin: the second-caller arm in ``pool.close()`` calls
    ``self._drain_remaining_after_cancel(deadline=...)`` (not the
    bare form).
    """
    src = _close_source()
    tree = ast.parse(src)
    found_with_deadline = False
    found_bare = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "_drain_remaining_after_cancel"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            if any(kw.arg == "deadline" for kw in node.keywords):
                found_with_deadline = True
            else:
                found_bare = True
    assert found_with_deadline, (
        "pool.close()'s second-caller arm must pass a ``deadline=`` "
        "to ``self._drain_remaining_after_cancel`` so the recovery "
        "sweep honours the close-budget envelope."
    )
    assert not found_bare, (
        "pool.close() should not contain a bare ``self._drain_remaining_after_cancel()`` call."
    )
