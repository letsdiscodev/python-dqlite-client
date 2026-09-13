"""_dial: TCP keepalive enabling and keepalive option failure handling."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import socket
import threading
from typing import Any

import pytest

from dqliteclient._dial import _apply_keepalive_options


def _serve_one_connection() -> tuple[socket.socket, int, threading.Event]:
    """Listen and accept exactly one connection; return (sock, port, event)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    accepted = threading.Event()

    def _accept_one() -> None:
        try:
            conn, _ = srv.accept()
            accepted.set()
            # Hold open so the client can inspect SO_KEEPALIVE before close.
            conn.recv(1)
            conn.close()
        except OSError:
            pass

    threading.Thread(target=_accept_one, daemon=True).start()
    return srv, port, accepted


@pytest.mark.asyncio
async def test_open_connection_helper_enables_so_keepalive() -> None:
    """The dial helper sets SO_KEEPALIVE=1 on the underlying socket."""
    from dqliteclient._dial import open_connection_with_keepalive

    srv, port, _ = _serve_one_connection()
    try:
        reader, writer = await asyncio.wait_for(
            open_connection_with_keepalive("127.0.0.1", port),
            timeout=5.0,
        )
        try:
            sock = writer.get_extra_info("socket")
            assert sock is not None, "expected a socket from the dial"
            keepalive = sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            assert keepalive == 1, (
                f"SO_KEEPALIVE not set on dialed socket; got {keepalive}. "
                "Half-open peers will not be detected within the kernel "
                "keepalive interval."
            )
            # TCP_NODELAY: without it, Nagle + delayed-ACK can stall a
            # small RPC up to ~40 ms. Mirrors Go's net.Dialer default.
            nodelay = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            assert nodelay == 1, (
                f"TCP_NODELAY not set on dialed socket; got {nodelay}. "
                "Nagle's algorithm will stall small RPCs."
            )
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
    finally:
        srv.close()


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
