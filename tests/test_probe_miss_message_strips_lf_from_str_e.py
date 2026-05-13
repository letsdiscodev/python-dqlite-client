"""``_ProbeMiss.message`` composition routes ``str(e)`` through
``sanitize_for_log`` so server-supplied LF / Tab in the wire-layer
exception text does not survive into the raised
``ClusterError.args[0]``.

The wire layer's ``sanitize_server_text`` deliberately preserves LF
and Tab so multi-line server diagnostics render correctly in exception
strings. But ``_probe_one`` aggregates per-node failure text into a
joined ``errors`` list that becomes the raised ``ClusterError``'s
message body — and any downstream code doing ``logger.exception(ce)``
or ``logger.error("%s", ce)`` would then split the operator's log
record on the LF (CWE-117 secondary path through exception text).
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


def test_lf_in_str_e_does_not_survive_into_cluster_error_args0() -> None:
    """A hostile peer's ``FailureResponse.message`` containing LF
    becomes part of the per-node failure exception text. The aggregate
    ClusterError text must not carry the raw LF — it must be escaped to
    literal ``\\n`` so a downstream ``logger.exception(ce)`` does not
    split the record."""
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
    # The escaped form survives as literal backslash-n so the forensic
    # context is still readable.
    assert "\\n" in raw_args0 or "forged second-line" in raw_args0


def test_tab_in_str_e_does_not_survive_into_cluster_error_args0() -> None:
    """The Tab vector is the second half of the sanitize_for_log
    discipline. Confirm it's escaped too."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=0.5)

    hostile = DqliteConnectionError("first\tcolumn\tinjected")
    with (
        patch.object(cc, "_query_leader", side_effect=hostile),
        pytest.raises(ClusterError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    raw_args0 = exc_info.value.args[0]
    assert "\t" not in raw_args0, f"raw Tab leaked into ClusterError.args[0]: {raw_args0!r}"
