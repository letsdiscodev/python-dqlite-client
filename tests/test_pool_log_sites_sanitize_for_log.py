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


def test_pool_logger_calls_sanitize_conn_address() -> None:
    """Every ``logger.*`` site in ``pool.py`` that references
    ``conn._address`` must wrap it via ``sanitize_for_log(str(...))``,
    not pass it raw."""
    source = _POOL_PY.read_text()
    offenders: list[str] = []
    for block in _logger_call_blocks(source):
        if "conn._address" not in block:
            continue
        # If the block mentions ``conn._address`` it must be inside
        # a ``sanitize_for_log(`` argument scope. A simple sanity
        # check: every occurrence of the substring ``conn._address``
        # must be preceded (within ~30 chars) by ``sanitize_for_log``.
        for occ in re.finditer(r"conn\._address", block):
            window = block[max(0, occ.start() - 64) : occ.start()]
            if "sanitize_for_log" not in window:
                offenders.append(block.splitlines()[0] + " ...")
                break
    assert not offenders, (
        "logger.* call(s) still pass raw conn._address; route through "
        "sanitize_for_log(str(conn._address)) for log-injection hygiene:\n" + "\n".join(offenders)
    )


def test_rollback_warning_sanitizes_exception_text() -> None:
    """The WARNING-level ROLLBACK-failure site interpolates the
    exception via ``%r``. Because exception ``__repr__`` echoes
    server-supplied text (``OperationalError(message)``), that text
    must be sanitised too — the warning passes
    ``sanitize_for_log(str(exc))`` rather than the raw ``exc``.
    """
    source = _POOL_PY.read_text()
    # Locate the WARNING site by message string, then walk balanced parens.
    needle = '"pool: dropping connection %s after ROLLBACK failure: %r"'
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


def test_pool_imports_sanitize_for_log_from_wire() -> None:
    """The import path must come from the public ``dqlitewire.messages.
    responses`` surface (post round-31 promotion), not the private
    ``_sanitize_*`` underscore name."""
    source = _POOL_PY.read_text()
    assert "from dqlitewire.messages.responses import sanitize_for_log" in source, (
        "pool.py must import sanitize_for_log from the public wire surface"
    )


def test_pool_initialize_warning_sanitizes_failure_repr() -> None:
    """``pool.initialize`` re-emits each per-create failure via a
    WARNING that interpolates the exception. Because exception
    ``__repr__`` echoes server-supplied ``OperationalError`` text
    (and ``DqliteConnectionError.args[0]``), that interpolation must
    pass through ``sanitize_for_log`` to strip control characters from
    hostile peer diagnostics."""
    source = _POOL_PY.read_text()
    needle = '"pool.initialize: create_connection %d/%d failed: %s"'
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
    assert "sanitize_for_log(repr(exc))" in block, (
        f"pool.initialize WARNING must wrap the failure exception via "
        f"sanitize_for_log(repr(exc)); block was:\n{block}"
    )


def test_pool_initialize_debug_sanitizes_first_failure_repr() -> None:
    """The companion DEBUG line at the abort-after-N-creates branch
    interpolates ``failures[0]`` and must apply the same discipline."""
    source = _POOL_PY.read_text()
    needle = '"closing %d survivors (first failure: %s)"'
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
    assert "sanitize_for_log(repr(failures[0]))" in block, (
        f"pool.initialize abort DEBUG must wrap failures[0] via "
        f"sanitize_for_log(repr(failures[0])); block was:\n{block}"
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
