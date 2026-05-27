"""Pin: ``_connection_unclosed_warning`` and ``_pool_unclosed_warning``
survive ``Py_FinalizeEx`` phase-3 module-globals-set-to-None teardown
without raising into the calling ``weakref.finalize`` machinery.

During interpreter shutdown ``PyImport_Cleanup`` walks ``sys.modules``
and sets module globals to ``None``. Without defensive capture, a
finalize body that re-reads those globals at call time would raise
``TypeError: 'NoneType' object is not callable`` and surface as an
unraisable-hook traceback at shutdown, drowning out the real cause.

The fix captures each module global as a kwarg-default at function-
definition time, mirroring ``_cleanup_loop_thread``'s discipline.

These tests verify the fix on two axes:

1. The body USES the captured kwarg-defaults (not re-read module
   globals): we replace the module global with a sentinel that
   raises on call. If the fix is wrong, the sentinel fires; if
   it's right, the captured default fires and the body emits
   normally.

2. The defensive early-bail branch handles the case where the
   captured default itself is somehow nulled: pass ``_x=None``
   explicitly and assert the body bails silently.
"""

from __future__ import annotations

import warnings as _stdlib_warnings

import pytest

from dqliteclient import connection as _conn_mod
from dqliteclient import pool as _pool_mod

_CREATOR_PID_SNAPSHOT = _conn_mod.get_current_pid()


class _RaisingSentinel:
    """Pretends to be a stdlib callable but raises if called or
    attribute-accessed. Used to prove a finalize body uses its
    captured kwarg-default rather than re-reading the module
    global."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"sentinel for {self.name} was called — body re-read module "
            "global instead of using its captured kwarg-default"
        )

    def __getattr__(self, item: str) -> object:
        raise AssertionError(
            f"sentinel for {self.name} had attribute {item!r} accessed — "
            "body re-read module global instead of using its captured "
            "kwarg-default"
        )


def _invoke_connection_finalizer(
    *, closed: bool = False, connected: bool = True, **kwargs: object
) -> None:
    _conn_mod._connection_unclosed_warning(
        closed_flag=[closed],
        connected_flag=[connected],
        address="127.0.0.1:9001",
        creator_pid=_CREATOR_PID_SNAPSHOT,
        **kwargs,
    )


def _invoke_pool_finalizer(
    *, closed: bool = False, reserved: bool = True, **kwargs: object
) -> None:
    _pool_mod._pool_unclosed_warning(
        closed_flag=[closed],
        reserved_flag=[reserved],
        queue=None,
        creator_pid=_CREATOR_PID_SNAPSHOT,
        **kwargs,
    )


# --- _connection_unclosed_warning — axis 1 (captured defaults) ---


def test_connection_body_survives_get_current_pid_set_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_current_pid`` is intentionally re-read at call time so
    the existing fork-pid tests' patch shim works. Shutdown safety
    comes from the broad ``except`` around the call site."""
    monkeypatch.setattr(_conn_mod, "get_current_pid", None)
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_connection_finalizer()


def test_connection_body_survives_get_current_pid_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raising() -> int:
        raise RuntimeError("phase-3 teardown sentinel")

    monkeypatch.setattr(_conn_mod, "get_current_pid", _raising)
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_connection_finalizer()


def test_connection_body_uses_captured_warnings_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_conn_mod, "warnings", _RaisingSentinel("warnings"))
    with pytest.warns(ResourceWarning, match="DqliteConnection"):
        _invoke_connection_finalizer()


def test_connection_body_uses_captured_contextlib_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_conn_mod, "contextlib", _RaisingSentinel("contextlib"))
    with pytest.warns(ResourceWarning, match="DqliteConnection"):
        _invoke_connection_finalizer()


def test_connection_body_uses_captured_sanitize_for_log_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_conn_mod, "sanitize_for_log", _RaisingSentinel("sanitize_for_log"))
    with pytest.warns(ResourceWarning, match="DqliteConnection"):
        _invoke_connection_finalizer()


# --- _connection_unclosed_warning — axis 2 (defensive bail) ---


def test_connection_bail_silently_when_warnings_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_connection_finalizer(_warnings=None)


def test_connection_bail_silently_when_contextlib_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_connection_finalizer(_contextlib=None)


def test_connection_bail_silently_when_sanitize_for_log_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_connection_finalizer(_sanitize_for_log=None)


# --- _connection_unclosed_warning — regression pin ---


def test_connection_emit_when_globals_intact() -> None:
    with pytest.warns(ResourceWarning, match="DqliteConnection"):
        _invoke_connection_finalizer(closed=False, connected=True)


# --- _pool_unclosed_warning — axis 1 ---


def test_pool_body_survives_get_current_pid_set_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_current_pid`` is re-read at call time so fork-pid
    tests' patch shim works. Shutdown safety comes from the broad
    ``except`` around the call site."""
    monkeypatch.setattr(_pool_mod, "get_current_pid", None)
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_pool_finalizer()


def test_pool_body_survives_get_current_pid_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raising() -> int:
        raise RuntimeError("phase-3 teardown sentinel")

    monkeypatch.setattr(_pool_mod, "get_current_pid", _raising)
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_pool_finalizer()


def test_pool_body_uses_captured_warnings_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_pool_mod, "warnings", _RaisingSentinel("warnings"))
    with pytest.warns(ResourceWarning, match="ConnectionPool"):
        _invoke_pool_finalizer()


def test_pool_body_uses_captured_contextlib_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_pool_mod, "contextlib", _RaisingSentinel("contextlib"))
    with pytest.warns(ResourceWarning, match="ConnectionPool"):
        _invoke_pool_finalizer()


# --- _pool_unclosed_warning — axis 2 ---


def test_pool_bail_silently_when_warnings_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_pool_finalizer(_warnings=None)


def test_pool_bail_silently_when_contextlib_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_pool_finalizer(_contextlib=None)


# --- _pool_unclosed_warning — regression pin ---


def test_pool_emit_when_globals_intact() -> None:
    with pytest.warns(ResourceWarning, match="ConnectionPool"):
        _invoke_pool_finalizer(closed=False, reserved=True)
