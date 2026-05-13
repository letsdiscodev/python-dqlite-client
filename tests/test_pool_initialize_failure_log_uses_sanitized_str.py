"""``ConnectionPool.initialize`` formats per-failure log records via
``f"{type(exc).__name__}: {sanitize_for_log(str(exc))}"`` rather than
``sanitize_for_log(repr(exc))``.

The repr-then-sanitise pipeline was injection-safe (repr() escapes LF /
CR / control bytes before the sanitiser sees them), but it produced
log lines like ``OperationalError('peer says \\x00\\u2028BAD', code=1)``
where the wire-layer single-stage substitution would produce the
cleaner ``OperationalError: peer says ?BAD``. This test pins the
operator-friendly form — class name still present, single-stage
sanitisation, no double-encoding.
"""

from __future__ import annotations

import logging

import pytest

from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_warning_does_not_double_escape_control_chars(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hostile peer text contains a NUL byte. The repr-then-sanitise
    pipeline would render it as ``\\x00`` (Python escape sequence); the
    str-then-sanitise pipeline replaces it with ``?`` (wire-layer
    single-stage substitution)."""
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
    # Class name must still be present in the log line.
    assert "OperationalError" in msg
    # NUL must be present as the wire-layer ``?`` substitution, not the
    # Python ``\\x00`` escape (which is the repr-then-sanitise shape).
    assert "\\x00" not in msg, (
        f"per-failure WARNING should use sanitize_for_log(str(exc)) not "
        f"sanitize_for_log(repr(exc)); got: {msg!r}"
    )


@pytest.mark.asyncio
async def test_initialize_warning_lf_in_peer_text_remains_escaped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LF in peer text must still be escaped (single line per record) —
    this is the CWE-117 invariant that survives the
    repr->str pipeline switch."""
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
