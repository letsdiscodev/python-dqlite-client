"""close() threads an aggregate close-budget deadline through both _drain_idle
and _drain_remaining_after_cancel so end-to-end wall-clock is bounded under a
stuck-close stampede (a bare _drain_idle() waits N x per_iter_cap)."""

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
    """pool.close() must call self._drain_idle(deadline=...), not the bare form."""
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
    """The cancel-recovery sweep must accept the same deadline so an outer cancel
    cannot re-amplify the gap via the recovery arm."""
    sig = _drain_remaining_after_cancel_signature()
    assert "deadline" in sig.parameters, (
        "_drain_remaining_after_cancel must accept a ``deadline`` "
        "kwarg so the cancel-recovery sweep honours the same "
        "aggregate close-budget the main drain loop enforces. "
        "Without it, an outer cancel re-amplifies the gap the "
        "main-loop deadline was meant to bound."
    )


def test_pool_close_forwards_deadline_into_drain_remaining_after_cancel() -> None:
    """The second-caller arm must call
    self._drain_remaining_after_cancel(deadline=...), not the bare form."""
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
