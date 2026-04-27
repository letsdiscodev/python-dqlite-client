"""Consolidated table-driven tests for ``_parse_savepoint_name`` and
``_parse_release_name``.

The parsers have several deliberate trade-off branches (None for
quoted / backtick / square-bracket / unicode / leading-digit names)
that historically each got their own ad-hoc tests as savepoint cycles
landed. This file consolidates the matrix so a future audit can extend
coverage with a single parametrize row instead of a new test method.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import _parse_release_name, _parse_savepoint_name


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        # Bare ASCII identifiers — accepted, lowercased.
        ("foo", "foo"),
        ("Foo", "foo"),
        ("FOO", "foo"),
        ("foo123", "foo123"),
        ("foo_bar", "foo_bar"),
        ("_foo", "_foo"),
        ("_", "_"),
        # Leading digit — parser-rejected (SQLite parse-rejects too).
        ("123foo", None),
        ("9", None),
        # Quoted / backtick / bracketed — parser-rejected to avoid the
        # case-sensitivity desync (server treats them case-sensitive,
        # bare branch lowercases).
        ('"foo"', None),
        ("`foo`", None),
        ("[foo]", None),
        # Unicode letters — parser-rejected; ``str.isalnum`` would
        # accept them but ``str.lower`` may normalise differently
        # than the server.
        ("café", None),
        ("αβγ", None),
        # Empty / whitespace-only — parser-rejected.
        ("", None),
        ("   ", None),
        ("\t\n", None),
        # Leading-comment + name — accepted (SQLite tokenizer treats
        # comments as whitespace anywhere in the token stream).
        ("/* x */ foo", "foo"),
        ("-- comment\nfoo", "foo"),
        # Trailing whitespace and comments — accepted.
        ("foo  ", "foo"),
        ("foo /* x */", "foo"),
        ("foo -- comment", "foo"),
        # Trailing garbage — parser-rejected (forward-defence
        # tightening; SQLite parse-rejects too).
        ("foo extra", None),
        ("foo()", None),
        ("foo bar", None),
    ],
)
def test_parse_savepoint_name(input_str: str, expected: str | None) -> None:
    assert _parse_savepoint_name(input_str) == expected


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        # Without leading SAVEPOINT keyword — delegates to _parse_savepoint_name.
        ("foo", "foo"),
        ("123foo", None),
        ('"foo"', None),
        # With leading SAVEPOINT keyword — keyword stripped before
        # delegating.
        ("SAVEPOINT foo", "foo"),
        ("SAVEPOINT  foo", "foo"),  # multiple spaces
        ("savepoint foo", "foo"),  # case-insensitive keyword
        ("SAVEPOINT\tfoo", "foo"),  # tab between keyword and name
        # SAVEPOINT keyword + comment + name.
        ("SAVEPOINT /* x */ foo", "foo"),
        ("SAVEPOINT -- comment\nfoo", "foo"),
        # ``SAVEPOINTX`` is a bareword, not the keyword — boundary
        # check rejects the strip; treats whole tail as a name (which
        # is itself a bareword starting with ``SAVEPOINTX foo`` — the
        # parser stops at the space, returns ``"savepointx"`` → no
        # wait, "SAVEPOINTX foo" has trailing garbage after
        # SAVEPOINTX, so parser-rejected after the trailing-garbage
        # tightening).
        ("SAVEPOINTX foo", None),
    ],
)
def test_parse_release_name(input_str: str, expected: str | None) -> None:
    assert _parse_release_name(input_str) == expected
