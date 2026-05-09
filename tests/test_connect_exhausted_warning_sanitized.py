"""Pin: ``ClusterClient.connect``'s all-attempts-exhausted WARNING
runs server-supplied text through ``sanitize_for_log`` before the
``logger.warning`` interpolation, matching the discipline already
applied to the find-leader aggregate WARNING.

Without this, a hostile peer that returns a ``FailureResponse`` with
``\\n`` / ``\\r`` in the message can split a single WARNING into
multiple log lines in syslog / journald / structured-log frontends.
The sibling find-leader aggregate WARNING at ``cluster.py`` already
applies ``sanitize_for_log``; the connect-exhausted WARNING was the
asymmetric outlier.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


def test_cluster_imports_sanitize_for_log_via_public_name() -> None:
    """Pin: ``dqliteclient.cluster`` imports the public ``sanitize_for_log``
    rather than the round-30 private ``_sanitize_for_log`` underscore
    name. Cross-package private imports were the round-31 cleanup
    target (parallel to ISSUE-1438 which promoted ``sanitize_server_text``).
    """
    from dqliteclient import cluster as cluster_mod

    assert hasattr(cluster_mod, "sanitize_for_log")
    # The wire-side public name resolves to the same callable.
    from dqlitewire import sanitize_for_log as wire_public

    assert cluster_mod.sanitize_for_log is wire_public


@pytest.mark.asyncio
async def test_connect_exhausted_warning_strips_lf_cr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dqliteclient.cluster")

    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=0.05)
    cluster._find_leader_impl = AsyncMock(
        side_effect=DqliteConnectionError(
            "leader unreachable\nINJECTED LOG LINE\rmore\nlines",
        )
    )

    with pytest.raises(DqliteConnectionError):
        await cluster.connect("main", max_attempts=1)

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dqliteclient.cluster"
    ]
    assert warnings, "exhausted-attempts WARNING must fire"
    msg = warnings[-1].getMessage()
    assert "connect exhausted" in msg
    assert "\n" not in msg, f"LF must be stripped from the WARNING: {msg!r}"
    assert "\r" not in msg, f"CR must be stripped from the WARNING: {msg!r}"
