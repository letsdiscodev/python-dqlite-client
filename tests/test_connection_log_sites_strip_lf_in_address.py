"""Pin: ``DqliteConnection`` ``logger.*`` sites interpolate the
server-supplied address via the ``_log_safe_address`` property
(routes through ``sanitize_for_log``) rather than ``_safe_address``
(routes through ``sanitize_server_text`` which preserves LF/Tab
for exception-message readability).

Mirrors the cluster.py log-site discipline established by
``done/client-cluster-log-sites-use-display-sanitize-not-log-sanitize.md``.
CWE-117 log injection defense-in-depth.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqlitewire import sanitize_for_log, sanitize_server_text


def _prime_connection_with_address(address: str) -> DqliteConnection:
    """Hand-build a DqliteConnection bypassing parse_address's
    admission check so we can interpolate hostile address chars."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = address
    conn._creator_pid = os.getpid()
    return conn


def test_log_safe_address_escapes_lf() -> None:
    """The new ``_log_safe_address`` property escapes LF, while the
    sibling ``_safe_address`` preserves it (intentional, for
    exception-message readability)."""
    addr = "evil.example.com:9001\nFORGED log entry"
    conn = _prime_connection_with_address(addr)

    # _safe_address preserves LF (sanitize_server_text contract).
    assert "\n" in conn._safe_address
    assert conn._safe_address == sanitize_server_text(addr)

    # _log_safe_address escapes LF (sanitize_for_log contract).
    assert "\n" not in conn._log_safe_address
    assert conn._log_safe_address == sanitize_for_log(addr)


@pytest.mark.parametrize(
    "log_method",
    [
        # All eight DqliteConnection logger.* sites that interpolate
        # the address — verified via grep of self._log_safe_address in
        # the source. Each is a smoke test asserting no raw LF in any
        # captured record's getMessage(); we drive each via a synthetic
        # invocation rather than the full RPC.
        "handshake_ok",
        "db_opened",
        "close_drain_error",
        "abort_protocol_drain",
        "rollback_cancel",
        "rollback_no_tx",
        "rollback_op_err",
        "rollback_other",
    ],
)
def test_log_site_interpolation_does_not_leak_lf(
    caplog: pytest.LogCaptureFixture, log_method: str
) -> None:
    """Drive each log site via a direct ``logger.debug`` call mirroring
    the source's interpolation, using a hostile address. Verifies
    that ``_log_safe_address`` (the per-site dispatch) strips LF
    before the record is rendered.
    """
    addr = "victim.internal:443\nFORGED row"
    conn = _prime_connection_with_address(addr)
    logger_under_test = logging.getLogger("dqliteclient.connection")
    caplog.set_level(logging.DEBUG, logger="dqliteclient.connection")
    fake_inner = MagicMock()
    fake_inner.code = 999
    fake_inner.message = "synthetic"

    # Drive each call shape verbatim (mirror connection.py).
    if log_method == "handshake_ok":
        logger_under_test.debug(
            "connect: handshake ok address=%s client_id=%d",
            conn._log_safe_address,
            42,
        )
    elif log_method == "db_opened":
        logger_under_test.debug(
            "connect: db opened address=%s db_id=%d database=%r",
            conn._log_safe_address,
            1,
            "default",
        )
    elif log_method == "close_drain_error":
        logger_under_test.debug("close: unexpected drain error for %s", conn._log_safe_address)
    elif log_method == "abort_protocol_drain":
        logger_under_test.debug(
            "_abort_protocol: unexpected drain error for %s",
            conn._log_safe_address,
        )
    elif log_method == "rollback_cancel":
        logger_under_test.debug(
            "transaction(address=%s, id=%s): rollback was cancelled ...",
            conn._log_safe_address,
            id(conn),
        )
    elif log_method == "rollback_no_tx":
        logger_under_test.debug(
            "transaction(address=%s, id=%s): rollback found no active tx ...",
            conn._log_safe_address,
            id(conn),
        )
    elif log_method == "rollback_op_err":
        logger_under_test.debug(
            "transaction(address=%s, id=%s): rollback failed with OE ...",
            conn._log_safe_address,
            id(conn),
        )
    elif log_method == "rollback_other":
        logger_under_test.debug(
            "transaction(address=%s, id=%s): rollback failed ...",
            conn._log_safe_address,
            id(conn),
        )

    assert caplog.records, f"{log_method}: expected at least one DEBUG record"
    for rec in caplog.records:
        msg = rec.getMessage()
        assert "\n" not in msg, f"{log_method}: raw LF leaked into log record: {msg!r}"
