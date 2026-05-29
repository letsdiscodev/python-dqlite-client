"""``ConnectionPool`` log sites must scrub peer-supplied text (``conn._address``,
ROLLBACK-failure exception text) through ``sanitize_for_log`` before logging, to
prevent log injection."""

from __future__ import annotations

import re
from pathlib import Path

_POOL_PY = Path(__file__).resolve().parent.parent / "src" / "dqliteclient" / "pool.py"


def _logger_call_blocks(source: str) -> list[str]:
    """Return the parenthesised argument text for every ``logger.<level>(...)`` call."""
    blocks: list[str] = []
    pattern = re.compile(r"logger\.(?:debug|info|warning|error|exception|critical)\(")
    for m in pattern.finditer(source):
        depth = 1
        i = m.end()
        while i < len(source) and depth > 0:
            ch = source[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        blocks.append(source[m.start() : i])
    return blocks


_ADDRESS_REFERENCE = re.compile(r"conn\._address|getattr\(\s*conn\s*,\s*['\"]_address['\"]")


def test_pool_logger_calls_sanitize_conn_address() -> None:
    """Every logger.* site referencing the connection address must wrap it via
    sanitize_for_log, not pass it raw."""
    source = _POOL_PY.read_text()
    offenders: list[str] = []
    for block in _logger_call_blocks(source):
        if not _ADDRESS_REFERENCE.search(block):
            continue
        for occ in _ADDRESS_REFERENCE.finditer(block):
            window = block[max(0, occ.start() - 64) : occ.start()]
            if "sanitize_for_log" not in window:
                offenders.append(block.splitlines()[0] + " ...")
                break
    assert not offenders, (
        "logger.* call(s) still pass raw connection address; route "
        "every ``conn._address`` AND ``getattr(conn, '_address', ...)`` "
        "site through sanitize_for_log(str(...)) for log-injection "
        "hygiene:\n" + "\n".join(offenders)
    )


def test_rollback_warning_sanitizes_exception_text() -> None:
    """The ROLLBACK-failure WARNING interpolates the exception via ``%s`` with the
    value pre-stringified through ``sanitize_for_log(repr(exc))``."""
    source = _POOL_PY.read_text()
    needle = '"pool: dropping connection %s after ROLLBACK failure: %s"'
    idx = source.find(needle)
    assert idx >= 0, (
        "Could not locate the pool-ROLLBACK-failure WARNING — test or production code drifted."
    )
    open_idx = source.rfind("logger.warning(", 0, idx)
    assert open_idx >= 0
    # Walk forward over balanced parens from the opening logger.warning(.
    i = open_idx + len("logger.warning(")
    depth = 1
    while i < len(source) and depth > 0:
        ch = source[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    block = source[open_idx:i]
    sanitize_calls = block.count("sanitize_for_log(")
    assert sanitize_calls >= 2, (
        f"WARNING block must sanitize both address and exc text; "
        f"found {sanitize_calls} sanitize_for_log(...) calls in:\n{block}"
    )
    assert "sanitize_for_log(repr(exc))" in block, (
        "ROLLBACK failure WARNING must wrap the exception via "
        "sanitize_for_log(repr(exc)) — same shape as the "
        "pool.initialize per-failure-warning."
    )


def test_pool_imports_sanitize_for_log_from_wire() -> None:
    """sanitize_for_log must be imported from the public dqlitewire surface
    (top-level or deep submodule), not a private underscore name."""
    source = _POOL_PY.read_text()
    top_level = "from dqlitewire import" in source and "sanitize_for_log" in source
    deep = "from dqlitewire.messages.responses import sanitize_for_log" in source
    assert top_level or deep, "pool.py must import sanitize_for_log from the public wire surface"


def test_pool_initialize_warning_sanitizes_failure_str() -> None:
    """pool.initialize re-emits each per-create failure via a WARNING that wraps
    str(exc) through sanitize_for_log, with class context via type(exc).__name__
    (repr-then-sanitise would double-encode)."""
    source = _POOL_PY.read_text()
    needle = '"pool.initialize: create_connection %d/%d failed: %s: %s"'
    idx = source.find(needle)
    assert idx >= 0, (
        "Could not locate the pool.initialize per-create-failure WARNING — production code drifted."
    )
    open_idx = source.rfind("logger.warning(", 0, idx)
    assert open_idx >= 0
    i = open_idx + len("logger.warning(")
    depth = 1
    while i < len(source) and depth > 0:
        ch = source[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    block = source[open_idx:i]
    assert "sanitize_for_log(str(exc))" in block, (
        f"pool.initialize WARNING must wrap the failure exception via "
        f"sanitize_for_log(str(exc)); block was:\n{block}"
    )
    assert "type(exc).__name__" in block, (
        f"pool.initialize WARNING must preserve class-context via "
        f"type(exc).__name__; block was:\n{block}"
    )


def test_pool_initialize_debug_sanitizes_first_failure_str() -> None:
    """The abort-branch DEBUG line wraps the first failure via sanitize_for_log(str(...))."""
    source = _POOL_PY.read_text()
    needle = '"closing %d survivors (first failure: %s: %s)"'
    idx = source.find(needle)
    assert idx >= 0, (
        "Could not locate the pool.initialize abort DEBUG line — production code drifted."
    )
    open_idx = source.rfind("logger.debug(", 0, idx)
    assert open_idx >= 0
    i = open_idx + len("logger.debug(")
    depth = 1
    while i < len(source) and depth > 0:
        ch = source[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    block = source[open_idx:i]
    assert "sanitize_for_log(str(_first_failure))" in block, (
        f"pool.initialize abort DEBUG must wrap the first failure via "
        f"sanitize_for_log(str(...)); block was:\n{block}"
    )
    assert "type(_first_failure).__name__" in block, (
        f"pool.initialize abort DEBUG must preserve class-context via "
        f"type(_first_failure).__name__; block was:\n{block}"
    )


def test_pool_log_sites_no_raw_exception_via_percent_r() -> None:
    """Every logger.* block using %r on a server-controlled exception identifier must
    wrap it through sanitize_for_log; local-enum/counter %r sites are out of scope."""
    source = _POOL_PY.read_text()
    server_exc_names = re.compile(r"\b(?:exc|exception|failure|failures|err|error)\b(?!\w)")
    offenders: list[str] = []
    for block in _logger_call_blocks(source):
        if "%r" not in block:
            continue
        if not server_exc_names.search(block):
            continue
        if "sanitize_for_log" not in block:
            offenders.append(block.splitlines()[0])
    assert not offenders, (
        "logger.* call(s) in pool.py use %r on a server-controlled "
        "exception without sanitize_for_log; route through "
        "sanitize_for_log(repr(...)) for log-injection hygiene:\n" + "\n".join(offenders)
    )


def test_synthetic_evil_address_does_not_inject_newline() -> None:
    """sanitize_for_log escapes an LF in an address to the literal ``\\n`` so a CRLF
    cannot split the log line."""
    from dqlitewire.messages.responses import sanitize_for_log

    out = sanitize_for_log("evil\nhost:9001")
    assert "\n" not in out, f"sanitize_for_log must escape LF; got {out!r}"
    assert "\\n" in out
