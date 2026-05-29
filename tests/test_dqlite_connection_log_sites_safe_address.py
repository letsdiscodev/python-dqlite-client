"""Pin: every ``logger.*`` call in ``DqliteConnection`` routes
``self._address`` through ``_safe_address`` so a future loosening of
``parse_address`` cannot re-open a CRLF / ANSI log-injection. Checked
statically by source inspection.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONNECTION_PY = Path(__file__).resolve().parent.parent / "src" / "dqliteclient" / "connection.py"


def _logger_call_blocks(source: str) -> list[str]:
    """Return the parenthesised arg-list text of every logger.<level>(...)
    call (best-effort balanced-paren scan)."""
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


def test_no_logger_call_uses_raw_self_address() -> None:
    source = _CONNECTION_PY.read_text()
    offenders: list[str] = []
    for block in _logger_call_blocks(source):
        if re.search(r"\bself\._address\b", block):
            offenders.append(block.splitlines()[0] + " ...")
    assert not offenders, (
        "logger.* call(s) still pass raw self._address; route through "
        "self._safe_address for defence-in-depth log-injection hygiene:\n" + "\n".join(offenders)
    )


def test_safe_address_property_routes_through_sanitiser() -> None:
    """Sanity: the _safe_address property exists (precondition for the
    static check above)."""
    from dqliteclient.connection import DqliteConnection

    descr = DqliteConnection.__dict__.get("_safe_address")
    assert descr is not None, (
        "DqliteConnection._safe_address property removed; the log-site "
        "sanitisation discipline relies on it."
    )
    assert isinstance(descr, property)


def test_synthetic_control_address_is_scrubbed_by_safe_address() -> None:
    """Functional pin: _safe_address scrubs CR / ANSI / bidi / zero-width
    to ``?`` (LF is deliberately left intact for multi-line diagnostics)."""
    from dqliteclient.connection import DqliteConnection

    # Bypass __init__ so parse_address doesn't reject the control bytes.
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = "evil:9001\r\x1b[31mANSI BIDI‮injected"

    safe = conn._safe_address
    assert "\r" not in safe, f"_safe_address must scrub CR; got {safe!r}"
    assert "\x1b" not in safe, f"_safe_address must scrub ESC; got {safe!r}"
    assert "‮" not in safe, f"_safe_address must scrub bidi RLO; got {safe!r}"
    assert "?" in safe
