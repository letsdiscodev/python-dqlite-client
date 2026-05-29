"""``_ProbeMiss.message`` routes ``str(e)`` through ``sanitize_for_log`` so
server-supplied LF/Tab cannot reach ``ClusterError.args[0]`` and split a
downstream ``logger.exception(ce)`` record (CWE-117)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


def test_lf_in_str_e_does_not_survive_into_cluster_error_args0() -> None:
    """LF in peer text must reach ClusterError.args[0] escaped, not raw."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=0.5)

    hostile = DqliteConnectionError("first line\nforged second-line")
    with (
        patch.object(cc, "_query_leader", side_effect=hostile),
        pytest.raises(ClusterError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    raw_args0 = exc_info.value.args[0]
    assert "\n" not in raw_args0, (
        f"raw LF leaked into ClusterError.args[0]; downstream "
        f"logger.exception(ce) would split the record into multiple log "
        f"lines. Got: {raw_args0!r}"
    )
    assert "\\n" in raw_args0 or "forged second-line" in raw_args0


def test_tab_in_str_e_does_not_survive_into_cluster_error_args0() -> None:
    """Tab vector: confirm it's escaped too."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=0.5)

    hostile = DqliteConnectionError("first\tcolumn\tinjected")
    with (
        patch.object(cc, "_query_leader", side_effect=hostile),
        pytest.raises(ClusterError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    raw_args0 = exc_info.value.args[0]
    assert "\t" not in raw_args0, f"raw Tab leaked into ClusterError.args[0]: {raw_args0!r}"
