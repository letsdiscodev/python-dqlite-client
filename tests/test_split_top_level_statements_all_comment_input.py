"""Pin: ``_split_top_level_statements`` returns the comment as a
single piece for all-comment input — downstream classifiers
(``_update_tx_flags_from_sql`` via ``_strip_leading_comments``) then
handle it as a no-op.

Existing tests cover ``""``, ``"   "``, ``";;"`` (all returning
empty list). The all-comment cases preserve the comment text since
the splitter only filters empty-after-strip pieces — comment text is
non-empty. Pin both behaviours so a future scanner refactor cannot
silently change either contract.
"""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


def test_line_comment_only_returns_single_piece() -> None:
    """``"-- comment\\n"`` is one piece (the comment text); downstream
    classifiers strip it via ``_strip_leading_comments``."""
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
