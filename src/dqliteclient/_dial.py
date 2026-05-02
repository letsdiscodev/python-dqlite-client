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
from typing import Final

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
    ``asyncio.open_connection`` returns; the only difference is the
    socket has SO_KEEPALIVE=1 (and best-effort
    TCP_KEEPIDLE/INTVL/CNT) applied before this function returns.
    """
    reader, writer = await asyncio.open_connection(host, port)
    sock = writer.get_extra_info("socket")
    if sock is not None:
        _apply_keepalive_options(sock)
    return reader, writer
