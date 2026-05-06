"""Pin: ``_run_protocol`` sets ``self._in_use = True`` INSIDE the
``try:`` block whose ``finally`` clears it.

A ``KeyboardInterrupt`` / ``SystemExit`` delivered at the bytecode
boundary BETWEEN the assignment and the ``try:`` (signals can be
delivered between any two bytecodes by the Python interpreter's
signal-eval-frequency machinery) escapes WITHOUT entering the try, so
the ``finally`` does not run and ``_in_use`` stays ``True`` forever.
Subsequent calls on the same ``DqliteConnection`` would raise
``InterfaceError("another operation is in progress")`` permanently —
the connection is unrecoverable until garbage-collected.

The dbapi sync facade has the symmetric pattern fixed (KI cleanup in
``_run_sync``); the client layer must match. The fix is one line —
move the ``self._in_use = True`` assignment to be the first statement
inside ``try:``.

Both pins (structural source-shape AND functional behaviour-via-
sys.settrace KI injection) defend the contract.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient import connection as _conn_mod
from dqliteclient.connection import DqliteConnection


def _make_connection() -> DqliteConnection:
    """Construct a minimal ``DqliteConnection`` whose state lets
    ``_run_protocol``'s preflight checks pass through to ``fn``.
    """
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = "host:9001"
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop_ref = None
    conn._pending_drain = None
    conn._creator_pid = _conn_mod._current_pid
    conn._pool_released = False
    conn._database = "main"
    conn._protocol = MagicMock()
    conn._db_id = 1
    # Suppress thread/loop binding so _check_in_use does not raise on
    # the first call from a synthetic test.
    conn._check_in_use = lambda: None  # type: ignore[method-assign,unused-ignore]
    return conn


def _get_run_protocol_source() -> str:
    """Read the dedented source of ``_run_protocol``."""
    src = inspect.getsource(DqliteConnection._run_protocol)
    return textwrap.dedent(src)


def test_in_use_assignment_is_first_statement_inside_try() -> None:
    """Source-level pin: ``self._in_use = True`` must be the first
    statement INSIDE the outermost ``try:`` of ``_run_protocol`` (the
    one whose ``finally`` clears the flag). A regression that moves
    the assignment back outside the ``try:`` (or any other ordering
    that places it before the try-frame is established) re-introduces
    the KI-leak window.
    """
    src = _get_run_protocol_source()
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.AsyncFunctionDef), (
        f"expected async function, got {type(func).__name__}"
    )

    # Find the outermost Try whose finally clears ``self._in_use = False``.
    target_try: ast.Try | None = None
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Try):
            continue
        for fin_stmt in stmt.finalbody:
            if not isinstance(fin_stmt, ast.Assign):
                continue
            if (
                len(fin_stmt.targets) == 1
                and isinstance(fin_stmt.targets[0], ast.Attribute)
                and fin_stmt.targets[0].attr == "_in_use"
                and isinstance(fin_stmt.value, ast.Constant)
                and fin_stmt.value.value is False
            ):
                target_try = stmt
                break
        if target_try is not None:
            break
    assert target_try is not None, (
        "expected a try/finally clearing self._in_use = False inside _run_protocol"
    )

    first = target_try.body[0]
    assert isinstance(first, ast.Assign), (
        f"first statement inside the try should be an Assign; got {type(first).__name__}"
    )
    assert (
        len(first.targets) == 1
        and isinstance(first.targets[0], ast.Attribute)
        and first.targets[0].attr == "_in_use"
        and isinstance(first.value, ast.Constant)
        and first.value.value is True
    ), (
        "first statement inside the _run_protocol try block must be "
        "``self._in_use = True``; placing it BEFORE the try re-introduces "
        "the KI-bytecode-boundary leak window"
    )


@pytest.mark.parametrize(
    "exc_cls",
    [KeyboardInterrupt, SystemExit],
)
@pytest.mark.asyncio
async def test_signal_class_exception_inside_fn_does_not_leak_in_use(
    exc_cls: type[BaseException],
) -> None:
    """Behavioural pin: a KI / SystemExit raised inside ``fn`` (the
    awaited body of the try) lands inside the existing
    ``except (CancelledError, KeyboardInterrupt, SystemExit)`` arm; the
    finally must clear ``_in_use``. Regression-fence the surrounding
    contract; the structural pin above is what defends the bytecode-
    boundary leak. The 2078-arm calls ``_invalidate(e)`` which is
    expected to clear ``_in_use`` itself, but the finally runs anyway.
    """
    conn = _make_connection()

    async def raiser(_protocol: Any, _db_id: int) -> None:
        raise exc_cls()

    with pytest.raises(exc_cls):
        await conn._run_protocol(raiser)
    assert conn._in_use is False, f"{exc_cls.__name__} escape must not leak _in_use=True"


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason="cannot inject via sys.settrace under coverage / debugger",
)
@pytest.mark.asyncio
async def test_kbi_injected_on_assignment_line_is_caught() -> None:
    """Behavioural pin via ``sys.settrace``: inject a
    ``KeyboardInterrupt`` exactly on the line that holds
    ``self._in_use = True``. After the fix, the assignment is the
    first line inside the try block, so the synthetic KI on that line
    lands INSIDE the try-frame and the finally runs. Before the fix,
    the assignment was outside the try-frame and the KI escaped without
    finally — leaving ``_in_use=True`` forever.

    This is the closest in-process repro of the "signal between two
    bytecodes" class — the trace hook fires per-line, simulating signal
    delivery on the assignment line specifically.
    """
    conn = _make_connection()

    src = _get_run_protocol_source()
    target_lineno_in_src = next(
        i + 1 for i, line in enumerate(src.splitlines()) if "self._in_use = True" in line
    )
    abs_lineno = DqliteConnection._run_protocol.__code__.co_firstlineno + target_lineno_in_src - 1
    target_filename = DqliteConnection._run_protocol.__code__.co_filename

    fired = {"injected": False}

    def trace_hook(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != target_filename:
            return None
        if event == "line" and frame.f_lineno == abs_lineno and not fired["injected"]:
            fired["injected"] = True
            raise KeyboardInterrupt("synthetic at-bytecode-boundary")
        return trace_hook

    async def noop_fn(_protocol: Any, _db_id: int) -> None:
        return None

    sys.settrace(trace_hook)
    try:
        with pytest.raises(KeyboardInterrupt):
            await conn._run_protocol(noop_fn)
    finally:
        sys.settrace(None)

    assert fired["injected"], "trace hook must have fired on the target line"
    assert conn._in_use is False, (
        "KI on the assignment line must not leak _in_use=True; the "
        "assignment must be the first line inside the try-block so "
        "the finally runs"
    )
