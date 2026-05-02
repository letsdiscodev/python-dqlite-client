"""Pin: every TCP dial in dqliteclient enables SO_KEEPALIVE.

CPython's ``asyncio.open_connection`` does not set SO_KEEPALIVE by
default; the OS default (Linux: keepalive disabled per-socket) leaves
half-open connections undetected until the kernel's tcp_retries2
budget elapses (~15 min on Linux). Go's net.Dialer enables 15s
keepalive by default, so the dqlite Go client gets this for free; the
Python client did not.

Pool's ``_socket_looks_dead`` peek catches the clean-FIN case; a
black-holed connection (k8s pod evicted without RST, NAT mapping
reaped) was not detected at all. Pin SO_KEEPALIVE=1 on every dial so
the kernel surfaces the dead peer within the keepalive interval.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading

import pytest


def _serve_one_connection() -> tuple[socket.socket, int, threading.Event]:
    """Start a TCP listener that accepts exactly one connection.
    Returns the listener socket, its bound port, and a stop-event."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    accepted = threading.Event()

    def _accept_one() -> None:
        try:
            conn, _ = srv.accept()
            accepted.set()
            # Hold the connection open briefly so the client side
            # can inspect SO_KEEPALIVE before close.
            conn.recv(1)
            conn.close()
        except OSError:
            pass

    threading.Thread(target=_accept_one, daemon=True).start()
    return srv, port, accepted


@pytest.mark.asyncio
async def test_open_connection_helper_enables_so_keepalive() -> None:
    """The shared dial helper must set SO_KEEPALIVE=1 on the
    underlying socket so half-open peers surface within the kernel
    keepalive interval rather than the multi-minute tcp_retries2
    budget."""
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
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
    finally:
        srv.close()
