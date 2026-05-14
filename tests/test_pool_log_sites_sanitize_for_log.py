"""Pin: ``ConnectionPool`` log sites scrub ``conn._address`` and the
ROLLBACK-failure exception text through ``sanitize_for_log`` before
display.

The cluster layer's ``done/ISSUE-F1`` / ``done/ISSUE-F2`` discipline
established that any peer-supplied or peer-derived text must pass
through ``sanitize_for_log`` (the LF-escaping wrapper around
``sanitize_server_text``) before reaching ``logger.*``. The pool
layer's four log sites at ``pool.py`` (``_socket_looks_dead`` drop,
``_is_no_tx_rollback_error`` debug, leader-class ROLLBACK failure,
generic ROLLBACK-failure WARNING) had been missed.

The WARNING at the generic-failure site is the highest-exposure: it
``%r``-formats the exception (which can echo server-supplied
``OperationalError`` text via ``__repr__``) AND interpolates the raw
``conn._address``. Both are now scrubbed.

This is a static-discipline pin: scan ``pool.py`` and assert that
every ``logger.*`` call referencing ``conn._address`` or the ROLLBACK
exception ``%r`` runs the value through ``sanitize_for_log``.
"""

from __future__ import annotations

import re
from pathlib import Path

_POOL_PY = Path(__file__).resolve().parent.parent / "src" / "dqliteclient" / "pool.py"


def _logger_call_blocks(source: str) -> list[str]:
    """Return source-text blocks for every ``logger.<level>(...)``
    call. Each block is the parenthesised argument list (best-effort
    balanced-paren scan)."""
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
    """Every ``logger.*`` site in ``pool.py`` that references the
    connection address — via either ``conn._address`` or
    ``getattr(conn, "_address", ...)`` — must wrap it via
    ``sanitize_for_log(str(...))``, not pass it raw. The pattern
    covers both shapes so a future site using either form is caught.
    """
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
    """The WARNING-level ROLLBACK-failure site interpolates the
    exception via ``%s`` with the value pre-stringified through
    ``sanitize_for_log(repr(exc))``. The earlier shape used ``%r`` on
    a sanitised string, doubly-quoting/escaping the operator-facing
    diagnostic; the canonical form (matching the
    ``pool.initialize`` per-failure-warning shape) is ``%s`` plus
    ``sanitize_for_log(repr(exc))``.
    """
    source = _POOL_PY.read_text()
    # Locate the WARNING site by message string, then walk balanced parens.
    needle = '"pool: dropping connection %s after ROLLBACK failure: %s"'
    idx = source.find(needle)
    assert idx >= 0, (
        "Could not locate the pool-ROLLBACK-failure WARNING — test or production code drifted."
    )
    # Walk back to the opening `logger.warning(`, then forward over balanced parens.
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
    # Must reference sanitize_for_log for the exc argument too.
    sanitize_calls = block.count("sanitize_for_log(")
    assert sanitize_calls >= 2, (
        f"WARNING block must sanitize both address and exc text; "
        f"found {sanitize_calls} sanitize_for_log(...) calls in:\n{block}"
    )
    # The exception interpolation must use repr() inside sanitize_for_log
    # so the OperationalError(...) shape reaches operators (vs. just
    # the bare message string).
    assert "sanitize_for_log(repr(exc))" in block, (
        "ROLLBACK failure WARNING must wrap the exception via "
        "sanitize_for_log(repr(exc)) — same shape as the "
        "pool.initialize per-failure-warning."
    )


def test_pool_imports_sanitize_for_log_from_wire() -> None:
    """The import path must come from the public ``dqlitewire`` top-level
    surface, not the private ``_sanitize_*`` underscore name. Either the
    deep submodule path or the top-level re-export is acceptable, since
    the top-level form (matching sibling sites in ``cluster.py`` and
    ``connection.py``) is preferred and the submodule path remains a
    public re-export."""
    source = _POOL_PY.read_text()
    top_level = "from dqlitewire import" in source and "sanitize_for_log" in source
    deep = "from dqlitewire.messages.responses import sanitize_for_log" in source
    assert top_level or deep, "pool.py must import sanitize_for_log from the public wire surface"


def test_pool_initialize_warning_sanitizes_failure_str() -> None:
    """``pool.initialize`` re-emits each per-create failure via a
    WARNING that interpolates ``str(exc)`` through ``sanitize_for_log``
    (previously ``repr(exc)``; the repr-then-sanitise pipeline was safe
    but double-encoded the same characters the wire-layer sanitiser
    was designed to handle — see
    ``done/pool-initialize-warning-uses-repr-not-sanitized-display.md``).
    The class-name context that ``repr()`` provided is preserved via a
    separate ``type(exc).__name__`` format arg."""
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
    """The companion DEBUG line at the abort-after-N-creates branch
    interpolates ``failures[0]`` via ``sanitize_for_log(str(...))`` (no
    longer ``repr``)."""
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
    """Targeted lint: every ``logger.*`` block in ``pool.py`` whose
    format string contains ``%r`` AND interpolates an identifier
    matching the server-controlled-exception pattern (``exc``,
    ``exception``, ``failure(s)``, ``err``) must wrap that
    interpolation through ``sanitize_for_log``. A future maintainer
    adding a new ``%r``-on-server-exception line would otherwise
    reintroduce the log-injection vector. Local-enum / row-counter
    ``%r`` sites (legitimate; not server-controlled) are out of scope.
    """
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
    """Functional pin: ``sanitize_for_log("evil\\nhost:9001")`` produces
    a single-line string with the LF escaped to the literal two-byte
    ``\\n`` sequence. Without this, a CRLF in the connection address
    would split the log line."""
    from dqlitewire.messages.responses import sanitize_for_log

    out = sanitize_for_log("evil\nhost:9001")
    assert "\n" not in out, f"sanitize_for_log must escape LF; got {out!r}"
    assert "\\n" in out
