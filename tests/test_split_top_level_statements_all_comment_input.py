"""``_split_top_level_statements`` returns all-comment input as a single
(non-empty) piece; downstream classifiers strip it to a no-op."""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


def test_line_comment_only_returns_single_piece() -> None:
    pieces = _split_top_level_statements("-- just a comment\n")
    assert pieces == ["-- just a comment"]


def test_block_comment_only_returns_single_piece() -> None:
    pieces = _split_top_level_statements("/* just a comment */")
    assert pieces == ["/* just a comment */"]


def test_multiple_comments_returns_single_piece() -> None:
    """No ``;`` boundary anywhere → single piece."""
    pieces = _split_top_level_statements("-- one\n/* two */-- three\n")
    assert pieces == ["-- one\n/* two */-- three"]


def test_comment_with_separator_splits() -> None:
    """A real ``;`` between comments still splits."""
    pieces = _split_top_level_statements("-- one\n; -- two\n")
    assert pieces == ["-- one", "-- two"]
