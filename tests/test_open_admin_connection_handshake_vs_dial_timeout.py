"""``ClusterClient.open_admin_connection`` distinguishes the dial-stall
arm from the handshake-stall arm with distinct error messages so an
operator's grep can land on the right diagnostic.

When ``attempt_timeout`` is smaller than the per-read ``timeout`` and
the handshake stalls (e.g. TLS-terminating sidecar with delayed
welcome), the outer envelope fires after the dial already returned.
A single generic ``Connection to <addr> timed out`` would mislead the
operator into suspecting the dial layer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_handshake_stall_uses_distinct_error_message() -> None:
    """A peer that completes dial quickly but stalls during the welcome
    handshake fires the outer ``attempt_timeout`` arm. The raised error
    must name 'handshake' so the operator can distinguish from the dial
    arm."""
    cc = ClusterClient(
        MemoryNodeStore(["127.0.0.1:9001"]),
        timeout=10.0,
        dial_timeout=5.0,
        attempt_timeout=0.05,
    )

    async def fake_open_connection(*_a: object, **_kw: object):
        # Return a (reader, writer) pair instantly — dial succeeds.
        reader = asyncio.StreamReader()
        # Build a minimal writer-ish stub with the methods the protocol
        # constructor / close path touches.

        class _Stub:
            def get_extra_info(self, *_a: object, **_kw: object) -> object:
                return None

            def is_closing(self) -> bool:
                return False

            def close(self) -> None:
                return None

            async def wait_closed(self) -> None:
                return None

            def write(self, *_a: object, **_kw: object) -> None:
                return None

            async def drain(self) -> None:
                return None

            def can_write_eof(self) -> bool:
                return False

            def transport(self) -> object:
                return None

        return reader, _Stub()

    async def stall_handshake(*_a: object, **_kw: object) -> None:
        # Sleep well past the outer attempt_timeout so the envelope
        # fires.
        await asyncio.sleep(5.0)

    with (
        patch("dqliteclient.cluster.open_connection", new=fake_open_connection),
        patch(
            # open_admin_connection migrated from the full handshake
            # to the lighter negotiate_protocol_only (go-parity);
            # stall the version-negotiation step instead.
            "dqliteclient.cluster.DqliteProtocol.negotiate_protocol_only",
            new=stall_handshake,
        ),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with cc.open_admin_connection("127.0.0.1:9001"):
            pass

    msg = str(exc_info.value)
    assert "handshake" in msg.lower(), (
        f"handshake-stall error should mention 'handshake' to distinguish "
        f"it from the dial-stall arm; got: {msg!r}"
    )
    assert "attempt_timeout" in msg, (
        f"handshake-stall error should mention 'attempt_timeout' for "
        f"operator-grep clarity; got: {msg!r}"
    )


@pytest.mark.asyncio
async def test_dial_stall_uses_legacy_connection_timed_out_message() -> None:
    """The dial-specific arm keeps the current 'Connection to ... timed
    out' message so existing operator runbooks / log queries continue to
    work."""
    cc = ClusterClient(
        MemoryNodeStore(["127.0.0.1:9001"]),
        timeout=10.0,
        dial_timeout=0.05,
        attempt_timeout=10.0,
    )

    async def stall_dial(*_a: object, **_kw: object):
        await asyncio.sleep(5.0)
        raise AssertionError("unreachable")

    with (
        patch("dqliteclient.cluster.open_connection", new=stall_dial),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with cc.open_admin_connection("127.0.0.1:9001"):
            pass

    msg = str(exc_info.value)
    assert "Connection to" in msg
    assert "timed out" in msg
    assert "handshake" not in msg.lower(), (
        f"dial-stall error must NOT mention handshake; got: {msg!r}"
    )
