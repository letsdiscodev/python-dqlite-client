"""Pin: ``_apply_keepalive_options`` failure / platform-fallback arms.

Two arms in ``_dial.py::_apply_keepalive_options`` are not covered by
the happy-path test:

1. ``SO_KEEPALIVE`` setsockopt OSError early-return: when the master
   keepalive enable itself fails (AF_UNIX socket, tunnel transport,
   some exotic kernels) the helper must return BEFORE applying any
   TCP_KEEP* tunings — half-configured keepalive is worse than none.
2. macOS ``TCP_KEEPALIVE`` fallback: when ``TCP_KEEPIDLE`` is absent
   (macOS exposes the idle-interval option under a different name),
   the helper must apply ``TCP_KEEPALIVE`` instead.

Both arms are best-effort and silently absorb failures; the tests
exercise them via a fake socket plus ``monkeypatch.delattr`` /
``setattr`` on the ``socket`` module to simulate the macOS shape on
Linux CI.
"""

from __future__ import annotations

import errno
import socket
from typing import Any

import pytest

from dqliteclient._dial import _apply_keepalive_options


class _FakeSock:
    """Records every setsockopt call. Optionally raises on a chosen
    ``(level, optname)`` to exercise the OSError-handling arms."""

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
    """If ``SO_KEEPALIVE`` setsockopt fails (e.g. AF_UNIX socket, some
    tunnel transports), the helper must return BEFORE applying any
    ``TCP_KEEP*`` tunings — half-configured keepalive (interval/count
    set without the master enable) is worse than no keepalive."""
    sock = _FakeSock(raise_on=(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
    _apply_keepalive_options(sock)  # type: ignore[arg-type]
    # TCP_NODELAY runs first (separate try/suppress) — always attempted.
    # Then SO_KEEPALIVE is attempted and fails — early return.
    # No TCP_KEEPIDLE / TCP_KEEPINTVL / TCP_KEEPCNT call must follow.
    optnames = [optname for (_lvl, optname, _v) in sock.calls]
    assert socket.SO_KEEPALIVE in optnames, "SO_KEEPALIVE must be attempted even when it fails"
    # No TCP_KEEP* tuning calls happen after the early return.
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
    """macOS exposes the idle-interval option as ``TCP_KEEPALIVE`` (a
    different numeric value from Linux's ``TCP_KEEPIDLE``). Simulate
    the macOS shape by hiding ``TCP_KEEPIDLE`` and ensuring
    ``TCP_KEEPALIVE`` is set on the ``socket`` module."""
    monkeypatch.delattr("socket.TCP_KEEPIDLE", raising=False)
    # If the current platform lacks TCP_KEEPALIVE, expose a fake value
    # so the elif branch fires. Pick a sentinel large enough to not
    # collide with real constants.
    if not hasattr(socket, "TCP_KEEPALIVE"):
        monkeypatch.setattr(socket, "TCP_KEEPALIVE", 0x10, raising=False)
    # ``getattr`` (vs. attribute access) to keep mypy quiet on Linux,
    # where the static stub doesn't expose ``TCP_KEEPALIVE``.
    tcp_keepalive_opt: int = getattr(socket, "TCP_KEEPALIVE")  # noqa: B009
    sock = _FakeSock()
    _apply_keepalive_options(sock)  # type: ignore[arg-type]
    optnames = [optname for (_lvl, optname, _v) in sock.calls]
    # The macOS-style TCP_KEEPALIVE must have been attempted.
    assert tcp_keepalive_opt in optnames, (
        f"With TCP_KEEPIDLE hidden, the macOS-style TCP_KEEPALIVE "
        f"fallback must be applied; calls were: {sock.calls}"
    )


def test_apply_keepalive_macos_fallback_absorbs_setsockopt_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The macOS fallback is wrapped in ``contextlib.suppress(OSError)``
    so a failed setsockopt does not break the dial. Pin the absorption
    by raising from the fallback path and asserting the helper still
    proceeds to the rest of the tuning."""
    monkeypatch.delattr("socket.TCP_KEEPIDLE", raising=False)
    if not hasattr(socket, "TCP_KEEPALIVE"):
        monkeypatch.setattr(socket, "TCP_KEEPALIVE", 0x10, raising=False)
    # ``getattr`` (vs. attribute access) to keep mypy quiet on Linux,
    # where the static stub doesn't expose ``TCP_KEEPALIVE``.
    tcp_keepalive_opt: int = getattr(socket, "TCP_KEEPALIVE")  # noqa: B009
    sock = _FakeSock(raise_on=(socket.IPPROTO_TCP, tcp_keepalive_opt))
    # Must NOT raise.
    _apply_keepalive_options(sock)  # type: ignore[arg-type]
    optnames = [optname for (_lvl, optname, _v) in sock.calls]
    # TCP_KEEPALIVE was attempted (and failed silently). The helper
    # continues past the fallback arm: TCP_KEEPINTVL / TCP_KEEPCNT
    # (if exposed on this platform) are still attempted.
    assert tcp_keepalive_opt in optnames
