"""CWE-117: ``_probe_one``'s per-node failure DEBUG log routes peer-supplied
exception text through ``sanitize_for_log`` so embedded LF/Tab can't forge
log records under a line-oriented collector."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterError,
    DqliteConnectionError,
)
from dqliteclient.node_store import MemoryNodeStore


def _find_probe_one_failure_debug_record(
    caplog: pytest.LogCaptureFixture,
) -> logging.LogRecord:
    for r in caplog.records:
        if (
            r.name == "dqliteclient.cluster"
            and r.levelno == logging.DEBUG
            and "find_leader" in r.msg
            and "failed with" in r.msg
        ):
            return r
    raise AssertionError(
        "expected a DEBUG record from _find_leader_impl._probe_one "
        "with 'find_leader: ... failed with ...' shape"
    )


def test_probe_one_failure_debug_log_escapes_lf_tab_in_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When ``_query_leader`` raises with LF/Tab in ``str(e)``, the DEBUG
    log arg must carry the escaped form (``\\n`` / ``\\t``)."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    store = MemoryNodeStore(["127.0.0.1:9001"])
    cc = ClusterClient(store, timeout=0.5, attempt_timeout=0.05)

    hostile_err = "connect refused\nMAY 14 12:34:56 leader-elected fake-node as leader\tcol2"

    with (
        patch.object(
            cc,
            "_query_leader",
            side_effect=DqliteConnectionError(hostile_err),
        ),
        pytest.raises(ClusterError),
    ):
        asyncio.run(cc.find_leader())

    rec = _find_probe_one_failure_debug_record(caplog)
    # rec.args: (address, type_name, exception_text_sanitized, idx, total)
    assert isinstance(rec.args, tuple)
    assert len(rec.args) == 5, f"expected 5 log args, got {rec.args!r}"
    exc_text_arg = rec.args[2]
    assert isinstance(exc_text_arg, str)
    assert "\n" not in exc_text_arg, (
        f"sanitize_for_log must escape LF in the exception-text arg "
        f"before the logger record; got raw LF in arg {exc_text_arg!r}"
    )
    assert "\t" not in exc_text_arg, (
        f"sanitize_for_log must escape Tab in the exception-text arg "
        f"before the logger record; got raw Tab in arg {exc_text_arg!r}"
    )
    assert "\\n" in exc_text_arg, (
        f"expected literal '\\n' escape in sanitized exception text; got {exc_text_arg!r}"
    )
    assert "\\t" in exc_text_arg, (
        f"expected literal '\\t' escape in sanitized exception text; got {exc_text_arg!r}"
    )
