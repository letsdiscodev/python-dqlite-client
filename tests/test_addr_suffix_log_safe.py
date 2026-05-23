"""Pin: ``DqliteProtocol._addr_suffix`` escapes LF/TAB via
``sanitize_for_log`` — not the display-only ``sanitize_server_text``
which preserves LF for multi-line exception readability.

The suffix flows into exception text that downstream
``logger.error("%s", exc)`` / ``logger.exception(...)`` sites
format; a peer-controlled LF in the address (from leader-redirect,
a malformed node store, or a dial_func override that bypassed
``parse_address``) would otherwise produce a journald record split
across rows (CWE-117).
"""

from __future__ import annotations

from dqliteclient.protocol import DqliteProtocol

_U2028 = " "


def _make_protocol_with_address(address: str) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = address
    return proto


def test_addr_suffix_escapes_lf() -> None:
    proto = _make_protocol_with_address("127.0.0.1:9001\nFAKE LOG")
    suffix = proto._addr_suffix()
    assert "\n" not in suffix, f"raw LF leaked into addr suffix: {suffix!r}"
    assert "\\n" in suffix, f"sanitize_for_log should escape LF; got {suffix!r}"


def test_addr_suffix_escapes_tab() -> None:
    proto = _make_protocol_with_address("127.0.0.1:9001\tFAKE LOG")
    suffix = proto._addr_suffix()
    assert "\t" not in suffix, f"raw TAB leaked into addr suffix: {suffix!r}"


def test_addr_suffix_strips_u2028() -> None:
    forged = f"127.0.0.1:9001{_U2028}FAKE LOG"
    proto = _make_protocol_with_address(forged)
    suffix = proto._addr_suffix()
    assert _U2028 not in suffix, f"U+2028 leaked into addr suffix: {suffix!r}"


def test_addr_suffix_empty_when_no_address() -> None:
    proto = _make_protocol_with_address("")
    assert proto._addr_suffix() == ""


def test_addr_suffix_safe_address_passes_through() -> None:
    """Negative pin: an already-safe address renders unchanged
    apart from the ' to ' prefix."""
    proto = _make_protocol_with_address("127.0.0.1:9001")
    suffix = proto._addr_suffix()
    assert suffix == " to 127.0.0.1:9001"
