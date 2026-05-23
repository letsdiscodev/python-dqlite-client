"""Pin: ``_query_leader``'s DEBUG log routes peer-supplied text
through ``sanitize_for_log`` — NOT ``_sanitize_display_text`` — so
LF and Tab are ESCAPED (``\\n`` / ``\\t``) at the logger boundary
rather than preserved verbatim.

The sibling site ``_verify_redirect`` already uses
``sanitize_for_log`` with an explicit code comment
calling out that ``_sanitize_display_text`` preserves LF / Tab for
exception-message readability — wrong helper for logger records
(CWE-117 log injection).

The threat: a hostile peer in a multi-node cluster returns a
``leader_address`` field with embedded LF / Tab. ``%r`` repr of
the value would coincidentally escape LF / Tab for that one
interpolation slot, but the discipline must hold at the sanitizer
boundary so the value is safe regardless of the conversion
specifier used downstream — `%s` slots in the same record carry
the same risk.

Test strategy: capture ``LogRecord.args`` (the raw, pre-format
arguments passed to the logger). On the old code path
``_sanitize_display_text`` preserves LF / Tab so the arg still
contains the raw byte; on the fixed code path
``sanitize_for_log`` replaces LF with the two-byte sequence
``\\n`` and Tab with ``\\t``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ProtocolError
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


def _fake_open_connection() -> object:
    async def _impl(host: str, port: int, **_kwargs: object) -> tuple[MagicMock, MagicMock]:
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    return _impl


def _find_query_leader_debug_record(
    caplog: pytest.LogCaptureFixture,
) -> logging.LogRecord:
    for r in caplog.records:
        if r.levelno == logging.DEBUG and "malformed redirect" in r.msg:
            return r
    raise AssertionError(
        "expected a DEBUG record from _query_leader with 'malformed redirect' in msg"
    )


@pytest.mark.asyncio
async def test_query_leader_zero_id_nonempty_addr_log_args_escape_lf_tab(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Arm 2 (``node_id == 0 and leader_addr``): the peer-supplied
    ``leader_addr`` carries LF / Tab. The logger must receive the
    escaped form (``\\n`` / ``\\t`` literals), proving the helper
    is ``sanitize_for_log`` and not ``_sanitize_display_text``."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock(return_value=10_000)
    fake_proto.negotiate_protocol_only = AsyncMock()
    hostile_leader_addr = "10.0.0.2:9001\nFAKE leader-elect record\tcol2"
    fake_proto.get_leader = AsyncMock(return_value=(0, hostile_leader_addr))

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.cluster"),
        patch("asyncio.open_connection", new=_fake_open_connection()),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(ProtocolError),
    ):
        await cluster._query_leader("localhost:9001", trust_server_heartbeat=False)

    rec = _find_query_leader_debug_record(caplog)
    # rec.args: (address_sanitized, leader_addr_sanitized)
    assert isinstance(rec.args, tuple)
    leader_addr_arg = rec.args[1]
    assert isinstance(leader_addr_arg, str)
    assert "\n" not in leader_addr_arg, (
        f"sanitize_for_log must escape LF before the logger record; "
        f"got raw LF in arg {leader_addr_arg!r}"
    )
    assert "\t" not in leader_addr_arg, (
        f"sanitize_for_log must escape Tab before the logger record; "
        f"got raw Tab in arg {leader_addr_arg!r}"
    )
    # Positive: the escape sequences are present as literal two-byte
    # forms (``\\n`` / ``\\t``).
    assert "\\n" in leader_addr_arg, (
        f"expected literal '\\n' escape in sanitized leader_addr; got {leader_addr_arg!r}"
    )
    assert "\\t" in leader_addr_arg, (
        f"expected literal '\\t' escape in sanitized leader_addr; got {leader_addr_arg!r}"
    )


@pytest.mark.asyncio
async def test_query_leader_nonzero_id_empty_addr_uses_sanitize_for_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Arm 1 (``node_id != 0 and not leader_addr``): the RAFT_NOMEM
    transient now returns ``None`` rather than raising — but it still
    emits a DEBUG breadcrumb naming the queried ``address``, and that
    must be routed through ``sanitize_for_log`` (not
    ``_sanitize_display_text``) to match the sibling discipline in
    ``_verify_redirect``."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock(return_value=10_000)
    fake_proto.negotiate_protocol_only = AsyncMock()
    # node_id != 0 with empty leader_addr triggers arm 1.
    fake_proto.get_leader = AsyncMock(return_value=(7, ""))

    from dqliteclient import cluster as cluster_mod

    real_sanitize_for_log = cluster_mod.sanitize_for_log  # type: ignore[attr-defined]
    real_display = cluster_mod._sanitize_display_text  # type: ignore[attr-defined]
    log_calls: list[str] = []
    display_calls: list[str] = []

    def spy_log(s: str) -> str:
        log_calls.append(s)
        return real_sanitize_for_log(s)

    def spy_display(s: str) -> str:
        display_calls.append(s)
        return real_display(s)

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.cluster"),
        patch.object(cluster_mod, "sanitize_for_log", new=spy_log),
        patch.object(cluster_mod, "_sanitize_display_text", new=spy_display),
        patch("asyncio.open_connection", new=_fake_open_connection()),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        result = await cluster._query_leader("localhost:9001", trust_server_heartbeat=False)
    assert result is None

    # The DEBUG breadcrumb routes the queried ``address`` through
    # ``sanitize_for_log``. Pin the helper identity so a future
    # refactor cannot regress into ``_sanitize_display_text`` (which
    # preserves LF / Tab — wrong for a logger record).
    assert log_calls, (
        "expected sanitize_for_log to be called inside the RAFT_NOMEM-transient DEBUG breadcrumb"
    )
