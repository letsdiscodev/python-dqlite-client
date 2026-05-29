"""``_connection_unclosed_warning`` / ``_pool_unclosed_warning`` survive
interpreter-shutdown teardown (module globals set to None) without raising
into the weakref.finalize machinery. They capture each module global as a
kwarg-default at definition time; tests cover both the captured-default use
and the defensive early-bail when a default is None.
"""

from __future__ import annotations

import warnings as _stdlib_warnings

import pytest

from dqliteclient import connection as _conn_mod
from dqliteclient import pool as _pool_mod

_CREATOR_PID_SNAPSHOT = _conn_mod.get_current_pid()


class _RaisingSentinel:
    """Raises if called or attribute-accessed, proving the finalize body
    uses its captured kwarg-default instead of re-reading the global."""

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


def test_connection_body_survives_get_current_pid_set_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_current_pid`` is re-read at call time (so the fork-pid tests'
    shim works); shutdown safety comes from the broad ``except``."""
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


def test_connection_emit_when_globals_intact() -> None:
    with pytest.warns(ResourceWarning, match="DqliteConnection"):
        _invoke_connection_finalizer(closed=False, connected=True)


def test_pool_body_survives_get_current_pid_set_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_current_pid`` is re-read at call time (so the fork-pid tests'
    shim works); shutdown safety comes from the broad ``except``."""
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


def test_pool_bail_silently_when_warnings_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_pool_finalizer(_warnings=None)


def test_pool_bail_silently_when_contextlib_kwarg_is_none() -> None:
    with _stdlib_warnings.catch_warnings():
        _stdlib_warnings.simplefilter("error", ResourceWarning)
        _invoke_pool_finalizer(_contextlib=None)


def test_pool_emit_when_globals_intact() -> None:
    with pytest.warns(ResourceWarning, match="ConnectionPool"):
        _invoke_pool_finalizer(closed=False, reserved=True)
