"""TCP dial helpers shared by the cluster and connection modules.

asyncio.open_connection leaves SO_KEEPALIVE off, so a black-holed peer
(no RST: pod evicted, NAT/firewall idle reap) is undetected until the
next write exhausts tcp_retries2 — a multi-minute hang. These helpers
force SO_KEEPALIVE=1 on every dial, with best-effort TCP_KEEP* tuning.
"""

import asyncio
import contextlib
import socket
from collections.abc import Awaitable, Callable
from typing import Final

# Caller-supplied dialer (go-dqlite ``WithDialFunc`` parity). Receives the
# full address string and owns ALL socket options (default keepalive/
# happy-eyeballs are bypassed). Transient faults must subclass OSError or
# TimeoutError to be retried; CancelledError must propagate.
type DialFunc = Callable[
    [str],
    Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
]

__all__ = [
    "DialFunc",
    "open_connection",
    "open_connection_with_keepalive",
]

# Best-effort per-socket keepalive tuning: idle 30s + 3 probes × 10s =
# ~60s to detect a black-holed peer. Applied only where the TCP_* options
# exist; the unconditional SO_KEEPALIVE=1 still applies everywhere.
_TCP_KEEPIDLE_S: Final[int] = 30
_TCP_KEEPINTVL_S: Final[int] = 10
_TCP_KEEPCNT: Final[int] = 3

# RFC 8305 happy-eyeballs IPv6/IPv4 fallback delay (Go's 300ms default).
# Without it, a dual-stack host with a broken AAAA pays the full TCP
# timeout before falling back to IPv4, doubling the effective dial_timeout.
_HAPPY_EYEBALLS_DELAY_S: Final[float] = 0.3
_HAPPY_EYEBALLS_INTERLEAVE: Final[int] = 1


def _apply_keepalive_options(sock: socket.socket) -> None:
    """Best-effort SO_KEEPALIVE/TCP_NODELAY/keepalive tuning; setsockopt
    failures are absorbed so an exotic platform cannot break the dial."""
    # TCP_NODELAY: dqlite RPCs are small and latency-sensitive; Nagle +
    # server delayed-ACK would stall each request up to ~40ms.
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        # Keepalive unavailable; dial still works, just without dead-peer detection.
        return
    if hasattr(socket, "TCP_KEEPIDLE"):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, _TCP_KEEPIDLE_S)
    elif hasattr(socket, "TCP_KEEPALIVE"):
        # macOS: TCP_KEEPALIVE is the idle-interval option (Linux's TCP_KEEPIDLE).
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
    """asyncio.open_connection with SO_KEEPALIVE and happy-eyeballs enabled."""
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
    """Dial via ``dial_func`` if given, else parse host:port and dial direct.

    The dial_func override receives the raw address and bypasses
    parse_address shape validation (CRLF/@/non-ASCII rejection), so a
    custom dialer must validate server-supplied redirect addresses itself.
    """
    if dial_func is not None:
        return await dial_func(address)
    # Function-scoped: connection imports from _dial, so avoid the cycle.
    from dqliteclient.connection import parse_address

    host, port = parse_address(address)
    return await open_connection_with_keepalive(host, port)
