"""Pin: ``DqliteConnection._connect_impl``'s rewrap arms route
peer-supplied exception text through ``_sanitize_display_text``
(the ``sanitize_server_text`` alias) before f-string interpolation.

CWE-117 defence-in-depth: a hostile peer's ``FailureResponse``
during the handshake can carry U+2028 LINE SEPARATOR / U+2029 /
bidi / zero-width inside the human-readable message field. The
display sanitiser strips those (replacing with ``?``) while
preserving LF / Tab for interactive-debug readability — the
project-wide ``_safe_address`` discipline.

Two rewrap arms in scope:

- ``LEADER_ERROR_CODES`` (``OperationalError`` arm) — wraps as
  ``DqliteConnectionError(f"Node {self._safe_address} is no longer
  leader: {e.message}", ...)``.
- ``WIRE_DECODE_FAILED_PREFIX`` (``ProtocolError`` arm) — wraps as
  ``DqliteConnectionError(f"{WIRE_DECODE_FAILED_PREFIX} during
  handshake to {self._safe_address}: {e}", ...)``.

``raw_message`` is unchanged (substring-matchers need verbatim
text); only the rendered ``str(exc)`` is hardened.
"""

from __future__ import annotations

import inspect

from dqliteclient.connection import DqliteConnection


def _strip_comments_and_docstrings(src: str) -> str:
    lines = []
    in_docstring = False
    for line in src.splitlines():
        stripped = line.strip()
        if in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') + stripped.count("'''") >= 2:
                continue
            in_docstring = True
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_leader_error_rewrap_sanitizes_e_message() -> None:
    """The LEADER_ERROR_CODES rewrap must interpolate
    ``_sanitize_display_text(e.message)`` rather than ``e.message``
    raw, so a peer-supplied U+2028 / bidi / ZW character cannot
    survive into ``str(DqliteConnectionError)``."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    # The rewrap must not contain the raw `{e.message}` interpolation
    # any more — it must route through the sanitiser.
    assert "{e.message}" not in code_only, (
        "LEADER_ERROR_CODES rewrap must sanitise ``e.message`` via "
        "_sanitize_display_text before interpolation; raw {e.message} "
        "would smuggle U+2028 / bidi / ZW into str(exc) and through "
        "to SA's is_disconnect / engine.echo logging."
    )
    assert "_sanitize_display_text(e.message)" in code_only, (
        "Expected the LEADER_ERROR_CODES rewrap to interpolate _sanitize_display_text(e.message)."
    )


def test_wire_decode_failed_rewrap_sanitizes_str_e() -> None:
    """The WIRE_DECODE_FAILED_PREFIX rewrap must interpolate
    ``_sanitize_display_text(str(e))`` rather than ``{e}`` raw."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "_sanitize_display_text(str(e))" in code_only, (
        "WIRE_DECODE_FAILED_PREFIX rewrap must sanitise ``str(e)`` "
        "via _sanitize_display_text before interpolation."
    )
