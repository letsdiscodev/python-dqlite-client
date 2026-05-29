"""Pin: every TCP dial enables SO_KEEPALIVE so black-holed peers
(no RST) surface within the keepalive interval, not the ~15 min
tcp_retries2 budget. asyncio.open_connection leaves it off by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading

import pytest


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
