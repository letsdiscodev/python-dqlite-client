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

__all__ = [
    "DialFunc",
    "open_connection",
    "open_connection_with_keepalive",
]

# Best-effort tunables for the per-socket keepalive interval. These
# do NOT match go-dqlite's ``net.Dialer{KeepAlive: 15s}`` defaults —
# Go sets both ``TCP_KEEPIDLE`` and ``TCP_KEEPINTVL`` to 15s and
# leaves ``TCP_KEEPCNT`` at the kernel default (9 on stock Linux per
# ``net.ipv4.tcp_keepalive_probes``), yielding a 15s first-probe
# horizon and ~150s worst-case detection budget. The Python values
# below pick a more conservative trade-off:
#
#   * 30s idle before the first probe (2× Go's 15s) — friendlier to
#     long-lived NAT mappings that drop idle entries after 30-60s.
#   * 10s between subsequent probes (vs Go's 15s) — faster retry
#     cadence once an idle peer is suspected.
#   * 3 probes before the kernel surfaces dead socket (vs Go's
#     ~9) — bounded probe budget.
#
# Net horizon: 30s + 10s × 3 = ~60s worst-case for a black-holed
# peer. This is ~2× SLOWER than Go for first-probe detection in the
# typical "dqlite cluster sees a peer restart" scenario, but ~2.5×
# FASTER than Go's worst-case 150s when every probe is lost.
# Operators tuning for cross-language deployments where peer-death
# latency must match Go should set ``_TCP_KEEPIDLE_S = 15``,
# ``_TCP_KEEPINTVL_S = 15``, ``_TCP_KEEPCNT = 9`` (or remove the
# Python-side override and inherit kernel defaults).
#
# Applied only on platforms that expose the corresponding ``TCP_*``
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
    """Enable SO_KEEPALIVE on ``sock``, disable Nagle's algorithm
    (TCP_NODELAY=1), and tune the per-socket keepalive interval
    where the platform supports it. Best-effort: failures on
    individual setsockopt calls are silently absorbed (they should
    never fire on Linux/macOS, but a kernel-level rejection on an
    exotic platform must not break the dial)."""
    # Disable Nagle's algorithm. Every dqlite RPC is small (<MSS) and
    # latency-sensitive; combined with delayed-ACK on the server,
    # Nagle stalls each request up to ~40 ms waiting for either an
    # ACK or buffered data to fill an MSS. Mirrors Go's net.Dialer
    # default (newTCPConn → setNoDelay(true)). AF_UNIX sockets and
    # other non-TCP transports ignore the option and may raise
    # ``OSError`` — silently absorb so the dial still succeeds.
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

    Admission-validation contract on the ``dial_func`` override path:
    the caller's dialer receives the raw ``address`` string. Default
    ``parse_address`` shape validation (CRLF / ``@`` /
    non-ASCII rejection) runs ONLY on the
    :func:`open_connection_with_keepalive` fallback path. Server-
    supplied redirect addresses pass through ``sanitize_server_text``
    at decode but NOT through ``parse_address`` on the dial_func
    override. Operators wrapping in TLS or binding to non-
    ``host:port`` transports (Unix socket, abstract socket) should
    validate the input shape inside the custom dialer. The non-
    ``host:port`` support on the override is deliberate — the helper
    cannot unconditionally call ``parse_address`` without breaking
    those operators. The downstream :class:`DqliteConnection.__init__`
    re-validates via ``parse_address`` and rejects malformed
    addresses before any wire RPC, but the user's dialer has already
    executed by that point. TLS likewise lives in the user's
    ``dial_func``: dqlite has no built-in TLS support, mirroring the
    Go/C client surface.
    """
    if dial_func is not None:
        return await dial_func(address)
    # Local import: ``connection`` imports from ``_dial`` so the
    # reverse import must stay function-scoped.
    from dqliteclient.connection import parse_address

    host, port = parse_address(address)
    return await open_connection_with_keepalive(host, port)
