"""Pin: ``_find_leader_impl``'s ``_probe_one`` per-node failure
DEBUG log routes the peer-supplied EXCEPTION TEXT through
``sanitize_for_log`` so embedded LF / Tab in ``str(e)`` are
ESCAPED (``\\n`` / ``\\t``) at the logger boundary rather than
preserved verbatim.

CWE-117 log injection. A hostile peer (or a buggy peer) whose
exception message contains a real newline lands verbatim in the
logger record on the old code path: under a line-oriented log
collector (journald, classic syslog, file-tailing pipelines), the
embedded LF is treated as a record boundary, so a peer returning
``"connect refused\\nMAY 14 12:34:56 leader-elected fake-node\\n"``
forges a fake "leader elected" log entry.

Sibling discipline: the ``_ProbeMiss`` return immediately below
the bad log site already wraps the same ``_truncate_error(str(e))``
in ``sanitize_for_log``. The defect class matches the existing
``_query_leader`` fix pinned by
``tests/test_query_leader_log_lf_tab_escape.py``.

Test strategy: capture ``LogRecord.args`` (the raw, pre-format
arguments passed to the logger). On the old code path the
exception-text arg still contains the raw LF / Tab bytes; on the
fixed code path ``sanitize_for_log`` replaces LF with the two-byte
sequence ``\\n`` and Tab with ``\\t``.
"""

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
    """``_probe_one`` per-node failure arm: when ``_query_leader``
    raises an exception whose ``str(e)`` contains LF / Tab, the
    DEBUG log site must thread the value through ``sanitize_for_log``
    so the logger record's arg has the escaped form (``\\n`` / ``\\t``
    literals)."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    store = MemoryNodeStore(["127.0.0.1:9001"])
    cc = ClusterClient(store, timeout=0.5, attempt_timeout=0.05)

    hostile_err = (
        "connect refused\n"
        "MAY 14 12:34:56 leader-elected fake-node as leader\t"
        "col2"
    )

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
    # Positive: the escape sequences are present as literal two-byte
    # forms (``\\n`` / ``\\t``).
    assert "\\n" in exc_text_arg, (
        f"expected literal '\\n' escape in sanitized exception text; "
        f"got {exc_text_arg!r}"
    )
    assert "\\t" in exc_text_arg, (
        f"expected literal '\\t' escape in sanitized exception text; "
        f"got {exc_text_arg!r}"
    )
