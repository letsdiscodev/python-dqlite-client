"""Table-driven tests for ``_parse_savepoint_name`` / ``_parse_release_name``."""

from __future__ import annotations

import pytest

from dqliteclient.connection import _parse_release_name, _parse_savepoint_name


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        # Bare ASCII — accepted, lowercased.
        ("foo", "foo"),
        ("Foo", "foo"),
        ("FOO", "foo"),
        ("foo123", "foo123"),
        ("foo_bar", "foo_bar"),
        ("_foo", "_foo"),
        ("_", "_"),
        # Leading digit — parser-rejected.
        ("123foo", None),
        ("9", None),
        # Quoted / backtick / bracketed — rejected: server is case-sensitive
        # but the bare branch lowercases (desync).
        ('"foo"', None),
        ("`foo`", None),
        ("[foo]", None),
        # Unicode — rejected: ``str.lower`` may normalise unlike the server.
        ("café", None),
        ("αβγ", None),
        # Empty / whitespace-only — rejected.
        ("", None),
        ("   ", None),
        ("\t\n", None),
        # Leading comment — accepted (tokenizer treats comments as whitespace).
        ("/* x */ foo", "foo"),
        ("-- comment\nfoo", "foo"),
        # Trailing whitespace / comments — accepted.
        ("foo  ", "foo"),
        ("foo /* x */", "foo"),
        ("foo -- comment", "foo"),
        # Trailing garbage — parser-rejected.
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
        # No leading keyword — delegates to _parse_savepoint_name.
        ("foo", "foo"),
        ("123foo", None),
        ('"foo"', None),
        # Leading SAVEPOINT keyword stripped before delegating.
        ("SAVEPOINT foo", "foo"),
        ("SAVEPOINT  foo", "foo"),
        ("savepoint foo", "foo"),  # case-insensitive keyword
        ("SAVEPOINT\tfoo", "foo"),
        ("SAVEPOINT /* x */ foo", "foo"),
        ("SAVEPOINT -- comment\nfoo", "foo"),
        # ``SAVEPOINTX`` is a bareword, not the keyword; trailing garbage
        # after it is parser-rejected.
        ("SAVEPOINTX foo", None),
    ],
)
def test_parse_release_name(input_str: str, expected: str | None) -> None:
    assert _parse_release_name(input_str) == expected
