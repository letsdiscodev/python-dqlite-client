"""``ConnectionPool.initialize`` formats per-failure log records via
``sanitize_for_log(str(exc))``, not ``sanitize_for_log(repr(exc))``, to avoid
double-encoding control bytes while keeping the class name present."""

from __future__ import annotations

import logging

import pytest

from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_warning_does_not_double_escape_control_chars(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A NUL byte in peer text renders as ``?`` (str-then-sanitise), not the
    ``\\x00`` Python escape (repr-then-sanitise)."""
    pool = ConnectionPool(["localhost:19001"], min_size=1, max_size=1, timeout=0.5)

    hostile = "peer\x00says BAD"

    async def _create_mock() -> object:
        raise OperationalError(hostile, code=1)

    pool._create_connection = _create_mock  # type: ignore[assignment]

    with (
        caplog.at_level(logging.WARNING, logger="dqliteclient.pool"),
        pytest.raises(OperationalError),
    ):
        await pool.initialize()

    warn_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "create_connection" in r.getMessage()
    ]
    assert warn_records, "expected a WARNING from pool.initialize per-failure log"
    msg = warn_records[0].getMessage()
    assert "OperationalError" in msg
    assert "\\x00" not in msg, (
        f"per-failure WARNING should use sanitize_for_log(str(exc)) not "
        f"sanitize_for_log(repr(exc)); got: {msg!r}"
    )


@pytest.mark.asyncio
async def test_initialize_warning_lf_in_peer_text_remains_escaped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LF in peer text stays escaped (single line per record): the CWE-117
    invariant survives the repr->str pipeline switch."""
    pool = ConnectionPool(["localhost:19001"], min_size=1, max_size=1, timeout=0.5)

    async def _create_mock() -> object:
        raise DqliteConnectionError("peer says\n[CRITICAL] forged log row")

    pool._create_connection = _create_mock  # type: ignore[assignment]

    with (
        caplog.at_level(logging.WARNING, logger="dqliteclient.pool"),
        pytest.raises(DqliteConnectionError),
    ):
        await pool.initialize()

    warn_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "create_connection" in r.getMessage()
    ]
    assert warn_records
    msg = warn_records[0].getMessage()
    assert "\n" not in msg, (
        f"WARNING record must remain single-line — sanitize_for_log "
        f"escapes LF to literal ``\\n``; got: {msg!r}"
    )
    assert "DqliteConnectionError" in msg
