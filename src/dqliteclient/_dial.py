"""TCP dial helpers shared between :mod:`dqliteclient.cluster` and
:mod:`dqliteclient.connection`.

The default ``asyncio.open_connection`` does not enable SO_KEEPALIVE on
the resulting socket. CPython inherits the OS default of "keepalive
disabled per-socket" (Linux: ``net.ipv4.tcp_keepalive_time = 7200`` is
irrelevant when the SO_KEEPALIVE option is off on the socket). Go's
``net.Dialer{}`` since Go 1.13 enables a 15s keepalive by default, so
the Go dqlite client gets dead-peer detection for free; the Python
client did not.

The pool's ``_socket_looks_dead`` peek catches the clean-FIN case (via
``transport.is_closing()`` and ``reader.at_eof()``) but not the
black-hole case where the peer disappears without RST (k8s pod evicted,
NAT mapping reaped, firewall idle expiry). Without keepalive the dead
socket is not detected until the kernel's ``tcp_retries2`` budget
elapses on the next write — multi-minute hang.

This helper enables SO_KEEPALIVE=1 unconditionally on every dial. The
optional TCP_KEEPIDLE / TCP_KEEPINTVL / TCP_KEEPCNT tuning is applied
under ``hasattr`` guards so the behaviour is best-effort across
platforms (Linux exposes all three; macOS uses ``TCP_KEEPALIVE`` for
the idle interval; Windows / BSD diverge further). The OS defaults
typically give a multi-minute interval, which is fine for surfacing a
dead peer faster than ``tcp_retries2`` without aggressive probing on
healthy idle connections.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import Awaitable, Callable
from typing import Final

# Caller-supplied dialer for go-dqlite parity with ``WithDialFunc``.
# A ``dial_func`` receives the FULL address string (opaque to
# dqliteclient — typically ``"host:port"`` but may be any shape the
# caller's dialer recognises) and returns the same ``(reader, writer)``
# pair :func:`asyncio.open_connection` returns.
#
# Contract:
#   * Caller-supplied dialer owns ALL socket options. The default
#     helper's SO_KEEPALIVE / TCP_KEEPIDLE / happy-eyeballs are
#     bypassed on the override path — operators wrapping in TLS or
#     binding to Unix sockets take responsibility for their own
#     keepalive / timeout discipline.
#   * Errors must subclass ``OSError`` or ``TimeoutError`` for the
#     existing per-call-site exception arms to recognise the failure
#     as a transient transport fault. A ``dial_func`` raising any
#     other class will surface unwrapped past the dial-arm except
#     blocks (acceptable: indicates dialer misconfig rather than a
#     transient peer fault).
#   * ``CancelledError`` must propagate; the helper is always awaited
#     inside ``asyncio.timeout(...)``.
type DialFunc = Callable[
    [str],
    Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
]

# Best-effort tunables for the per-socket keepalive interval. These
# match the defaults Go's ``net.Dialer`` uses when KeepAlive is set to
# 15s (its 1.13+ default). We pick more conservative values to avoid
# disturbing healthy idle-but-NAT-friendly connections: 30s idle before
# the first probe, 10s between probes, 3 probes before declaring dead
# (= ~60s detection horizon for a black-holed peer). These values are
# applied only on platforms that expose the corresponding ``TCP_*``
# socket options; the unconditional SO_KEEPALIVE=1 still applies
# everywhere.
_TCP_KEEPIDLE_S: Final[int] = 30
_TCP_KEEPINTVL_S: Final[int] = 10
_TCP_KEEPCNT: Final[int] = 3

# RFC 8305 happy-eyeballs delay between launching the IPv6 attempt and
# the IPv4 fallback attempt. Matches Go's ``net.Dialer{FallbackDelay:
# 300ms}`` default since Go 1.12. On a dual-stack hostname where the
# AAAA record points to an unroutable address (legitimate misconfig:
# legacy ``::ffff:`` mapping or a mis-set ALIAS), serial fallback would
# pay the full TCP timeout on the AAAA before falling back to A —
# making a 5s ``dial_timeout`` take 10s+ on Python where Go takes 5s.
# Python's ``asyncio.open_connection`` accepts ``happy_eyeballs_delay``
# (and ``interleave``) since 3.10; the package requires >= 3.13 so
# both are unconditionally available.
_HAPPY_EYEBALLS_DELAY_S: Final[float] = 0.3
_HAPPY_EYEBALLS_INTERLEAVE: Final[int] = 1


def _apply_keepalive_options(sock: socket.socket) -> None:
    """Enable SO_KEEPALIVE on ``sock`` and tune the per-socket
    keepalive interval where the platform supports it. Best-effort:
    failures on individual setsockopt calls are silently absorbed
    (they should never fire on Linux/macOS, but a kernel-level
    rejection on an exotic platform must not break the dial)."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        # Cannot enable keepalive at all — fall through; the dial
        # still works, just without the dead-peer-detection improvement.
        return
    if hasattr(socket, "TCP_KEEPIDLE"):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, _TCP_KEEPIDLE_S)
    elif hasattr(socket, "TCP_KEEPALIVE"):
        # macOS: TCP_KEEPALIVE is the idle-interval option (numeric
        # value differs from Linux's TCP_KEEPIDLE).
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, _TCP_KEEPIDLE_S)
    if hasattr(socket, "TCP_KEEPINTVL"):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, _TCP_KEEPINTVL_S)
    if hasattr(socket, "TCP_KEEPCNT"):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, _TCP_KEEPCNT)


async def open_connection_with_keepalive(
    host: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Wrap :func:`asyncio.open_connection` to enable SO_KEEPALIVE on
    the dialed socket.

    Returns the same ``(reader, writer)`` tuple
    ``asyncio.open_connection`` returns; the differences are:

    1. The socket has SO_KEEPALIVE=1 (and best-effort
       TCP_KEEPIDLE/INTVL/CNT) applied before this function returns.
    2. RFC 8305 happy-eyeballs (``happy_eyeballs_delay=0.3``,
       ``interleave=1``) is enabled, matching Go's
       ``net.Dialer{FallbackDelay: 300ms}`` default since Go 1.12.
       On dual-stack hostnames with broken AAAA, this avoids waiting
       the full TCP timeout before falling back to IPv4.
    """
    reader, writer = await asyncio.open_connection(
        host,
        port,
        happy_eyeballs_delay=_HAPPY_EYEBALLS_DELAY_S,
        interleave=_HAPPY_EYEBALLS_INTERLEAVE,
    )
    sock = writer.get_extra_info("socket")
    if sock is not None:
        _apply_keepalive_options(sock)
    return reader, writer


async def open_connection(
    address: str,
    *,
    dial_func: DialFunc | None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Dial ``address`` via a caller-supplied :data:`DialFunc` if
    provided, otherwise fall through to
    :func:`open_connection_with_keepalive` after parsing
    ``host:port`` out of the address.

    Mirrors go-dqlite's ``Connector`` dial dispatch (default
    ``net.Dialer{}.DialContext`` overridden by ``WithDialFunc``). The
    custom-dial path bypasses every default socket option this module
    sets on the TCP path; the caller owns SO_KEEPALIVE / TLS / etc.
    """
    if dial_func is not None:
        return await dial_func(address)
    # Local import: ``connection`` imports from ``_dial`` so the
    # reverse import must stay function-scoped.
    from dqliteclient.connection import parse_address

    host, port = parse_address(address)
    return await open_connection_with_keepalive(host, port)
