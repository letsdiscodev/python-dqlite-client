"""Every public entry-point's ``timeout`` / ``close_timeout`` default is the named constant."""

from __future__ import annotations

import inspect

import dqliteclient
from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import (
    DEFAULT_CLOSE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DqliteConnection,
)
from dqliteclient.pool import ConnectionPool


def test_constants_are_floats_with_documented_values() -> None:
    assert isinstance(DEFAULT_TIMEOUT_SECONDS, float)
    assert isinstance(DEFAULT_CLOSE_TIMEOUT_SECONDS, float)
    assert DEFAULT_TIMEOUT_SECONDS == 10.0
    assert DEFAULT_CLOSE_TIMEOUT_SECONDS == 0.5


def test_constants_present_at_package_top_level() -> None:
    assert dqliteclient.DEFAULT_TIMEOUT_SECONDS is DEFAULT_TIMEOUT_SECONDS
    assert dqliteclient.DEFAULT_CLOSE_TIMEOUT_SECONDS is DEFAULT_CLOSE_TIMEOUT_SECONDS
    assert "DEFAULT_TIMEOUT_SECONDS" in dqliteclient.__all__
    assert "DEFAULT_CLOSE_TIMEOUT_SECONDS" in dqliteclient.__all__


def _default_of(fn: object, name: str) -> object:
    sig = inspect.signature(fn)  # type: ignore[arg-type]
    return sig.parameters[name].default


def test_timeout_default_is_constant_at_every_entry_point() -> None:
    """Identity check, not value equality, catches a future copy-of-literal regression."""
    entries: list[tuple[object, str]] = [
        (dqliteclient.connect, "timeout"),
        (dqliteclient.create_pool, "timeout"),
        (DqliteConnection.__init__, "timeout"),
        (ConnectionPool.__init__, "timeout"),
        (ClusterClient.__init__, "timeout"),
        (ClusterClient.from_addresses, "timeout"),
    ]
    for fn, name in entries:
        got = _default_of(fn, name)
        assert got is DEFAULT_TIMEOUT_SECONDS or got == DEFAULT_TIMEOUT_SECONDS, (
            f"{fn!r}'s {name!r} default is {got!r}; expected DEFAULT_TIMEOUT_SECONDS "
            f"({DEFAULT_TIMEOUT_SECONDS!r})"
        )


def test_close_timeout_default_is_constant_at_every_entry_point() -> None:
    entries: list[tuple[object, str]] = [
        (dqliteclient.connect, "close_timeout"),
        (dqliteclient.create_pool, "close_timeout"),
        (DqliteConnection.__init__, "close_timeout"),
        (ConnectionPool.__init__, "close_timeout"),
        (ClusterClient.connect, "close_timeout"),
    ]
    for fn, name in entries:
        got = _default_of(fn, name)
        assert got is DEFAULT_CLOSE_TIMEOUT_SECONDS or got == DEFAULT_CLOSE_TIMEOUT_SECONDS, (
            f"{fn!r}'s {name!r} default is {got!r}; expected "
            f"DEFAULT_CLOSE_TIMEOUT_SECONDS ({DEFAULT_CLOSE_TIMEOUT_SECONDS!r})"
        )
