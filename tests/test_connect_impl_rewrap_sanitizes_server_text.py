"""CWE-117: ``_connect_impl``'s rewrap arms route peer-supplied exception text
through ``_sanitize_display_text`` before f-string interpolation so a hostile
peer cannot smuggle U+2028 / bidi / ZW into ``str(exc)``. ``raw_message`` stays
verbatim for substring-matchers; only the rendered ``str(exc)`` is hardened."""

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
    """The LEADER_ERROR_CODES rewrap interpolates
    ``_sanitize_display_text(e.message)``, not raw ``e.message``."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
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
    """The WIRE_DECODE_FAILED_PREFIX rewrap interpolates
    ``_sanitize_display_text(str(e))``, not raw ``{e}``."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    code_only = _strip_comments_and_docstrings(src)
    assert "_sanitize_display_text(str(e))" in code_only, (
        "WIRE_DECODE_FAILED_PREFIX rewrap must sanitise ``str(e)`` "
        "via _sanitize_display_text before interpolation."
    )
