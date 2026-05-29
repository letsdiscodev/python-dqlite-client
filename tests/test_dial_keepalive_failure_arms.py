"""Pin: ``_apply_keepalive_options`` failure / platform-fallback arms
(SO_KEEPALIVE setsockopt early-return, macOS TCP_KEEPALIVE fallback).
"""

from __future__ import annotations

import errno
import socket
from typing import Any

import pytest

from dqliteclient._dial import _apply_keepalive_options


class _FakeSock:
    """Records setsockopt calls; optionally raises on a chosen (level, optname)."""

    def __init__(
        self,
        raise_on: tuple[int, int] | None = None,
        raise_exc: type[OSError] = OSError,
    ) -> None:
        self.calls: list[tuple[int, int, Any]] = []
        self._raise_on = raise_on
        self._raise_exc = raise_exc

    def setsockopt(self, level: int, optname: int, value: Any) -> None:
        self.calls.append((level, optname, value))
        if self._raise_on is not None and (level, optname) == self._raise_on:
            raise self._raise_exc(errno.ENOPROTOOPT, "fake setsockopt failure")


def test_apply_keepalive_returns_early_on_so_keepalive_failure() -> None:
    """On SO_KEEPALIVE setsockopt failure the helper returns before any
    TCP_KEEP* tuning — half-configured keepalive is worse than none."""
    sock = _FakeSock(raise_on=(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
    _apply_keepalive_options(sock)  # type: ignore[arg-type]
    optnames = [optname for (_lvl, optname, _v) in sock.calls]
    assert socket.SO_KEEPALIVE in optnames, "SO_KEEPALIVE must be attempted even when it fails"
    keepidle = getattr(socket, "TCP_KEEPIDLE", None)
    keepintvl = getattr(socket, "TCP_KEEPINTVL", None)
    keepcnt = getattr(socket, "TCP_KEEPCNT", None)
    tcp_keepalive_macos = getattr(socket, "TCP_KEEPALIVE", None)
    forbidden = {x for x in (keepidle, keepintvl, keepcnt, tcp_keepalive_macos) if x is not None}
    leaked = forbidden.intersection(optnames)
    assert not leaked, (
        f"TCP_KEEP* options must NOT be applied after SO_KEEPALIVE "
        f"failure (half-configured keepalive); leaked: {sorted(leaked)}"
    )


def test_apply_keepalive_uses_macos_tcp_keepalive_when_keepidle_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With TCP_KEEPIDLE hidden (macOS shape), the helper applies the
    macOS TCP_KEEPALIVE idle-interval option instead."""
    monkeypatch.delattr("socket.TCP_KEEPIDLE", raising=False)
    if not hasattr(socket, "TCP_KEEPALIVE"):
        monkeypatch.setattr(socket, "TCP_KEEPALIVE", 0x10, raising=False)
    # getattr (not attribute access) keeps mypy quiet on Linux, where
    # the static stub omits TCP_KEEPALIVE.
    tcp_keepalive_opt: int = getattr(socket, "TCP_KEEPALIVE")  # noqa: B009
    sock = _FakeSock()
    _apply_keepalive_options(sock)  # type: ignore[arg-type]
    optnames = [optname for (_lvl, optname, _v) in sock.calls]
    assert tcp_keepalive_opt in optnames, (
        f"With TCP_KEEPIDLE hidden, the macOS-style TCP_KEEPALIVE "
        f"fallback must be applied; calls were: {sock.calls}"
    )


def test_apply_keepalive_macos_fallback_absorbs_setsockopt_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed setsockopt in the macOS fallback is absorbed (the dial
    must not break)."""
    monkeypatch.delattr("socket.TCP_KEEPIDLE", raising=False)
    if not hasattr(socket, "TCP_KEEPALIVE"):
        monkeypatch.setattr(socket, "TCP_KEEPALIVE", 0x10, raising=False)
    # getattr (not attribute access) keeps mypy quiet on Linux, where
    # the static stub omits TCP_KEEPALIVE.
    tcp_keepalive_opt: int = getattr(socket, "TCP_KEEPALIVE")  # noqa: B009
    sock = _FakeSock(raise_on=(socket.IPPROTO_TCP, tcp_keepalive_opt))
    _apply_keepalive_options(sock)  # type: ignore[arg-type]
    optnames = [optname for (_lvl, optname, _v) in sock.calls]
    assert tcp_keepalive_opt in optnames
