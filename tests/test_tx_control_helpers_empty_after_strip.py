"""Pin: ``_starts_with_tx_verb`` and
``_sql_is_outermost_release_or_commit`` return ``False`` on
all-comment / whitespace-only input.

Both predicates strip leading SQL comments and whitespace then
return ``False`` if the result is empty. This is currently exercised
indirectly via ``_split_top_level_statements`` (see
``test_split_top_level_statements_all_comment_input.py``), but the
predicates themselves have no direct pin. A future refactor that
tightened the empty-after-strip case to an exception or to ``True``
would ripple silently through both call sites.
"""

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
    """Sanity: the strip-then-match path must still recognise a verb
    when one survives the strip."""
    assert _starts_with_tx_verb("/* hint */ BEGIN") is True


def _make_connection() -> DqliteConnection:
    """Connection for direct method invocation. We never call
    ``connect()``, so the underlying socket is never opened — only
    the predicate's pure-string path is exercised."""
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
