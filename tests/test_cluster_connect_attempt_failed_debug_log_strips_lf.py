"""Pin: ``ClusterClient.connect``'s per-attempt-failure DEBUG log
sanitises LF / Tab in the interpolated exception via
``sanitize_for_log(_truncate_error(str(exc)))``. Mirrors the
sibling WARNING at attempts-exhausted (cluster.py:1551-1556).

A hostile peer can stuff ``\\n`` into a server-returned message
(``sanitize_server_text`` deliberately preserves LF for interactive
exception readability). The DEBUG arm was the lone log site at
the connect-retry call surface that interpolated the raw exception
``%s`` — CWE-117 log injection at DEBUG level.
"""

from __future__ import annotations

import logging

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_connect_attempt_failed_debug_log_strips_lf_in_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Drive a synthetic ``DqliteConnectionError`` carrying LF in its
    message into the connect-retry loop; verify the per-attempt
    DEBUG record does not split into multiple journald lines.
    """
    addr_with_lf = "victim:9001"
    poisoned_message = "leader-msg\nFORGED log row"
    cluster = ClusterClient(MemoryNodeStore([addr_with_lf]))

    async def _exploding_find_leader(*, trust_server_heartbeat: bool = False, policy=None) -> str:
        raise DqliteConnectionError(poisoned_message)

    monkeypatch.setattr(cluster, "find_leader", _exploding_find_leader)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
    # retry_with_backoff re-raises the inner exception after exhausting
    # retries (it's already a retryable type); the DEBUG records get
    # written on every attempt regardless.
    with pytest.raises((ClusterError, DqliteConnectionError)):
        await cluster.connect()

    debug_records = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and "ClusterClient.connect attempt" in r.getMessage()
        and "failed" in r.getMessage()
    ]
    assert debug_records, "expected one or more 'attempt N/M failed' DEBUG records from connect()"
    for rec in debug_records:
        msg = rec.getMessage()
        assert "\n" not in msg, (
            f"per-attempt DEBUG log leaked raw LF from server exception message: {msg!r}"
        )
