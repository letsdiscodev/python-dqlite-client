"""Lexical helpers for SQL text: comment stripping, statement splitting, literal blanking.

These are lexical only (no grammar). They exist so the client's transaction
tracker and the dbapi's statement classifier share one tokenizer.
"""

import re
import string
from typing import Final

__all__ = [
    "blank_literals_and_comments",
    "is_keyword_boundary",
    "leading_keyword",
    "split_statements",
    "strip_leading_comments",
]

_IDENT_REST: Final[frozenset[str]] = frozenset(string.ascii_letters + string.digits + "_")

_LITERALS_AND_COMMENTS_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    '(?:[^']|'')*'          # single-quoted string literal
    | "(?:[^"]|"")*"        # double-quoted identifier
    | \[[^\]]*\]            # bracket-quoted identifier
    | `(?:[^`]|``)*`        # backtick-quoted identifier
    | --[^\n]*              # line comment
    | /\*.*?\*/             # block comment
    """,
    re.VERBOSE | re.DOTALL,
)


def is_keyword_boundary(s: str, kw_len: int) -> bool:
    """True if position ``kw_len`` in ``s`` ends an SQL word (``_`` is a word char)."""
    return len(s) == kw_len or s[kw_len] not in _IDENT_REST


def strip_leading_comments(sql: str) -> str:
    """Strip leading whitespace, a UTF-8 BOM, and ``--`` / ``/* */`` comments."""
    s = sql.lstrip("﻿").strip()
    while True:
        if s.startswith("--"):
            newline = s.find("\n")
            if newline == -1:
                return ""
            s = s[newline + 1 :].strip()
        elif s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                return ""
            s = s[end + 2 :].strip()
        else:
            return s


def leading_keyword(sql: str) -> str:
    """Upper-cased first word of ``sql`` after leading comments and ``(``; ``""`` if none."""
    s = strip_leading_comments(sql).lstrip("( \t\r\n")
    end = 0
    while end < len(s) and _is_word_char(s[end]):
        end += 1
    return s[:end].upper()


def blank_literals_and_comments(sql: str) -> str:
    """Replace string literals, quoted identifiers and comments with a single space."""
    return _LITERALS_AND_COMMENTS_RE.sub(" ", sql)


def split_statements(sql: str) -> list[str]:
    """Split on top-level ``;``, returning stripped non-empty pieces.

    Skips ``;`` inside literals, quoted identifiers and comments, and inside a
    ``CREATE [TEMP] TRIGGER ... BEGIN ... END`` body.
    """
    out: list[str] = []
    start = 0
    i = 0
    n = len(sql)
    in_trigger_body = False
    trigger_depth = 0
    case_depth = 0
    trigger_scan_start = 0
    while i < n:
        c = sql[i]
        if c in "'\"[`":
            i = _skip_quoted(sql, i, n)
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c.isalpha() and (i == 0 or not _is_word_char(sql[i - 1])):
            kw_end = i
            while kw_end < n and _is_word_char(sql[kw_end]):
                kw_end += 1
            kw = sql[i:kw_end].upper()
            if not in_trigger_body:
                if kw == "CREATE" and i >= trigger_scan_start:
                    j = _scan_for_trigger_begin(sql, kw_end, n)
                    if j > 0:
                        in_trigger_body = True
                        trigger_depth = 1
                        i = j
                        continue
            elif kw == "BEGIN":
                trigger_depth += 1
            elif kw == "CASE":
                case_depth += 1
            elif kw == "END":
                if case_depth > 0:
                    case_depth -= 1
                else:
                    trigger_depth -= 1
                    if trigger_depth == 0:
                        in_trigger_body = False
            i = kw_end
            continue
        if c == ";" and not in_trigger_body:
            piece = sql[start:i].strip()
            if piece:
                out.append(piece)
            start = i + 1
            trigger_scan_start = start
        i += 1
    tail = sql[start:].strip()
    if tail:
        out.append(tail)
    return out


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _skip_quoted(sql: str, i: int, n: int) -> int:
    """Return the index just past the quoted token starting at ``sql[i]``."""
    quote = sql[i]
    close = "]" if quote == "[" else quote
    i += 1
    while i < n:
        if sql[i] == close:
            if close != "]" and i + 1 < n and sql[i + 1] == close:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _skip_ws_and_comments(sql: str, i: int, n: int) -> int:
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
        elif c == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
        elif c == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            break
    return i


def _scan_for_trigger_begin(sql: str, after_create: int, n: int) -> int:
    """From just after ``CREATE``, return the index past a trigger body's ``BEGIN``, else 0."""
    i = _skip_ws_and_comments(sql, after_create, n)
    j = i
    while j < n and _is_word_char(sql[j]):
        j += 1
    word = sql[i:j].upper()
    if word in ("TEMP", "TEMPORARY"):
        i = _skip_ws_and_comments(sql, j, n)
        j = i
        while j < n and _is_word_char(sql[j]):
            j += 1
        word = sql[i:j].upper()
    if word != "TRIGGER":
        return 0
    i = j
    paren_depth = 0
    while i < n:
        c = sql[i]
        if c in "'\"[`":
            i = _skip_quoted(sql, i, n)
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth = max(0, paren_depth - 1)
        elif c == ";":
            return 0
        elif paren_depth == 0 and c.isalpha() and (i == 0 or not _is_word_char(sql[i - 1])):
            j = i
            while j < n and _is_word_char(sql[j]):
                j += 1
            if sql[i:j].upper() == "BEGIN":
                return j
            i = j
            continue
        i += 1
    return 0
