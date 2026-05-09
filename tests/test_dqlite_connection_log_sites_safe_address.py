"""Pin: every ``logger.*`` call inside ``DqliteConnection`` routes
``self._address`` through ``_safe_address`` (i.e. the wire-layer
``_sanitize_display_text`` helper), so a future loosening of
``parse_address`` cannot silently re-open a CRLF / ANSI log-injection
on the connection-layer log surface.

Background:

- ``_safe_address`` (``connection.py``) pipes ``self._address`` through
  ``dqlitewire.messages.responses.sanitize_for_log``-equivalent
  scrubbing before display.
- Every exception-raise site already used ``_safe_address``;
  the parallel ``logger.*`` sites used the raw ``self._address``.
- ``parse_address`` rejects whitespace / CRLF today, so the production
  exposure is theoretical — but the discipline is the established
  pattern for cross-layer log-site sanitisation (sibling siblings
  ``done/ISSUE-F1`` / ``done/ISSUE-F2`` / ``done/client-dqlite-
  connection-self-address-not-sanitised-in-exception-fstrings.md``).

We don't need to actually drive the connection through every cited
log call — the discipline can be pinned mechanically by source
inspection: every ``logger.*`` call in ``connection.py`` that mentions
``self.<addr>`` must reference ``self._safe_address``, never the raw
``self._address``.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONNECTION_PY = Path(__file__).resolve().parent.parent / "src" / "dqliteclient" / "connection.py"


def _logger_call_blocks(source: str) -> list[str]:
    """Return source-text blocks for every ``logger.<level>(...)``
    call. Each block is the parenthesised argument list (best-effort
    balanced-paren scan, sufficient for this static check).
    """
    blocks: list[str] = []
    pattern = re.compile(r"logger\.(?:debug|info|warning|error|exception|critical)\(")
    for m in pattern.finditer(source):
        # Walk forward over balanced parentheses.
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
    """Every ``logger.*`` site in ``connection.py`` that references
    ``self.<address>`` must use ``self._safe_address``, not raw
    ``self._address``."""
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
    """Sanity: the ``_safe_address`` property exists and delegates to
    the wire-layer ``_sanitize_display_text``. The presence of the
    property is the precondition for the static check above."""
    from dqliteclient.connection import DqliteConnection

    # The descriptor is a property on the class.
    descr = DqliteConnection.__dict__.get("_safe_address")
    assert descr is not None, (
        "DqliteConnection._safe_address property removed; the log-site "
        "sanitisation discipline relies on it."
    )
    assert isinstance(descr, property)


def test_synthetic_control_address_is_scrubbed_by_safe_address() -> None:
    """Functional pin: a synthetic ``self._address`` containing
    control / bidi characters is scrubbed by ``_safe_address`` before
    reaching the log formatter. ``_safe_address`` delegates to the
    wire-layer ``sanitize_server_text`` which replaces CR / ANSI / bidi
    / zero-width codepoints with ``?``. (It deliberately leaves LF
    intact so legitimate multi-line server diagnostics render in
    exception messages — log-injection via LF specifically is closed
    at ``parse_address`` today, plus the higher-volume log surfaces
    use ``sanitize_for_log`` directly.)
    """
    from dqliteclient.connection import DqliteConnection

    # Build a minimal instance bypassing __init__ (so ``parse_address``
    # doesn't reject the synthetic control bytes up-front).
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = "evil:9001\r\x1b[31mANSI BIDI‮injected"

    safe = conn._safe_address
    # CR, ESC (\x1b), and the U+202E bidi override must all be replaced.
    assert "\r" not in safe, f"_safe_address must scrub CR; got {safe!r}"
    assert "\x1b" not in safe, f"_safe_address must scrub ESC; got {safe!r}"
    assert "‮" not in safe, f"_safe_address must scrub bidi RLO; got {safe!r}"
    assert "?" in safe
