"""``_starts_with_tx_verb`` and ``_sql_is_outermost_release_or_commit``
return ``False`` on all-comment / whitespace-only input."""

from __future__ import annotations

from dqliteclient import DqliteConnection
from dqliteclient.connection import _starts_with_tx_verb


def test_starts_with_tx_verb_returns_false_for_pure_dash_dash_comment() -> None:
    assert _starts_with_tx_verb("-- comment\n") is False


def test_starts_with_tx_verb_returns_false_for_pure_block_comment() -> None:
    assert _starts_with_tx_verb("/* x */") is False


def test_starts_with_tx_verb_returns_false_for_whitespace_only() -> None:
    assert _starts_with_tx_verb("   ") is False


def test_starts_with_tx_verb_returns_false_for_empty_string() -> None:
    assert _starts_with_tx_verb("") is False


def test_starts_with_tx_verb_recognises_begin_after_comments() -> None:
    """The strip-then-match path still recognises a surviving verb."""
    assert _starts_with_tx_verb("/* hint */ BEGIN") is True


def _make_connection() -> DqliteConnection:
    """Unconnected connection; only the predicate's pure-string path runs."""
    conn = DqliteConnection("127.0.0.1:9001")
    return conn


def test_sql_is_outermost_release_or_commit_false_for_pure_dash_dash_comment() -> None:
    conn = _make_connection()
    assert conn._sql_is_outermost_release_or_commit("-- comment\n") is False


def test_sql_is_outermost_release_or_commit_false_for_pure_block_comment() -> None:
    conn = _make_connection()
    assert conn._sql_is_outermost_release_or_commit("/* x */") is False


def test_sql_is_outermost_release_or_commit_false_for_whitespace_only() -> None:
    conn = _make_connection()
    assert conn._sql_is_outermost_release_or_commit("   ") is False


def test_sql_is_outermost_release_or_commit_false_for_empty_string() -> None:
    conn = _make_connection()
    assert conn._sql_is_outermost_release_or_commit("") is False
