"""``_update_tx_flags_from_sql`` SAVEPOINT / RELEASE / ROLLBACK TO tracking.

A bare ``SAVEPOINT name`` outside a transaction triggers SQLite's implicit
BEGIN (mirrors stdlib sqlite3); SAVEPOINT inside an explicit BEGIN is a no-op
on the outer boundary.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestSavepointAutobeginTracking:
    def test_bare_savepoint_starts_transaction(self, conn: DqliteConnection) -> None:
        assert conn._in_transaction is False
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        assert conn._in_transaction is True

    def test_release_of_outermost_autobegin_savepoint_ends_transaction(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT sp1")
        assert conn._in_transaction is False

    def test_release_without_savepoint_keyword_also_ends_transaction(
        self, conn: DqliteConnection
    ) -> None:
        # SQLite accepts both ``RELEASE name`` and ``RELEASE SAVEPOINT name``.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("RELEASE sp1")
        assert conn._in_transaction is False

    def test_savepoint_inside_explicit_begin_does_not_change_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT sp1")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("COMMIT")
        assert conn._in_transaction is False

    def test_nested_savepoint_release_of_inner_keeps_outer_active(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT inner")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT outer")
        assert conn._in_transaction is False

    def test_rollback_to_savepoint_does_not_change_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        # ROLLBACK TO leaves the named savepoint active; boundary unchanged.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("ROLLBACK TO SAVEPOINT sp1")
        assert conn._in_transaction is True

    def test_rollback_to_unwinds_intermediate_frames(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT mid")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        # ROLLBACK TO outer: frames above outer are removed; outer stays.
        conn._update_tx_flags_from_sql("ROLLBACK TO SAVEPOINT outer")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT outer")
        assert conn._in_transaction is False

    def test_plain_rollback_clears_savepoint_state(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._in_transaction is False

    def test_release_of_known_to_server_unknown_to_client_clears_stack(
        self, conn: DqliteConnection
    ) -> None:
        # RELEASE of a name absent from the local stack: the server pops
        # the named frame plus every frame above it, and we cannot know
        # which tracked frames sat above. Clear and set the untracked flag
        # so the pool-reset safety net fires.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT does_not_exist")
        assert conn._savepoint_stack == []
        assert conn._has_untracked_savepoint is True

    def test_rollback_to_known_to_server_unknown_to_client_marks_untracked(
        self, conn: DqliteConnection
    ) -> None:
        # ROLLBACK TO does not pop the target (only frames above), so
        # clearing the stack could drop frames below the unobserved target.
        # Leave the stack alone; set the untracked flag.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("ROLLBACK TO SAVEPOINT does_not_exist")
        assert conn._savepoint_stack == ["sp1"]
        assert conn._has_untracked_savepoint is True

    def test_savepoint_with_double_quoted_name_not_tracked(self, conn: DqliteConnection) -> None:
        # Quoted names are case-sensitive but the unquoted branch lowercases;
        # tracking quoted forms would let a later unquoted RELEASE collide
        # and pop the stack while the server refuses. No-op for quoted names.
        conn._update_tx_flags_from_sql('SAVEPOINT "weird name"')
        assert conn._in_transaction is False
        assert conn._savepoint_stack == []

    def test_savepoint_quoted_then_release_unquoted_does_not_pop_stack(
        self, conn: DqliteConnection
    ) -> None:
        # Quoted SAVEPOINT is not tracked, so a later unquoted RELEASE
        # finds nothing on the stack and is a no-op.
        conn._update_tx_flags_from_sql('SAVEPOINT "MyPoint"')
        conn._update_tx_flags_from_sql("RELEASE mypoint")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_tracked_savepoint_after_untracked_does_not_claim_implicit_begin(
        self, conn: DqliteConnection
    ) -> None:
        # Autobegin happened on the outer untracked frame, not on inner;
        # promoting inner to the autobegin frame would let a later RELEASE
        # inner flip _in_transaction=False while the server still holds the tx.
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        assert conn._savepoint_stack == ["inner"]
        assert conn._savepoint_implicit_begin is False
        assert conn._has_untracked_savepoint is True

    def test_in_transaction_property_true_after_quoted_savepoint(
        self, conn: DqliteConnection
    ) -> None:
        # The property reports True when the server auto-begun a tx even
        # though the local stack cannot model the frame (matches stdlib
        # sqlite3 and the pool-reset predicate).
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._in_transaction is False
        assert conn._has_untracked_savepoint is True
        assert conn.in_transaction is True

    def test_release_inner_tracked_after_untracked_outer_keeps_untracked_flag(
        self, conn: DqliteConnection
    ) -> None:
        # Untracked outer, tracked inner, RELEASE inner: an empty tracked
        # stack must not flip _in_transaction False while the outer
        # autobegun tx is still server-side alive.
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("RELEASE inner")
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False
        assert conn._has_untracked_savepoint is True

    def test_savepoint_quoted_then_release_quoted_same_text_no_op(
        self, conn: DqliteConnection
    ) -> None:
        # Even a matching-text quoted RELEASE is not tracked, because the
        # SAVEPOINT itself was not pushed.
        conn._update_tx_flags_from_sql('SAVEPOINT "MyPoint"')
        conn._update_tx_flags_from_sql('RELEASE "MyPoint"')
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_savepoint_name_case_insensitive_match(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT MyPoint")
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT mypoint")
        assert conn._in_transaction is False


class TestSavepointStateClearedOnInvalidate:
    def test_invalidate_clears_savepoint_stack(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        assert conn._savepoint_stack
        conn._invalidate()
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False
        assert conn._in_transaction is False

    async def test_close_clears_savepoint_stack(self) -> None:
        conn = DqliteConnection("localhost:9001")
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._pool_released = False
        await conn.close()
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False


class TestSavepointNameParserBareIdentifierShape:
    """``_parse_savepoint_name`` accepts only ASCII alphanumerics with a
    leading non-digit; non-ASCII folds differently from the server's, so
    reject up front to keep the local stack in step."""

    def test_leading_digit_returns_none(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("1foo") is None

    def test_leading_underscore_accepted(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("_sp") == "_sp"

    def test_unicode_letter_returns_none(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("αβγ") is None

    def test_unicode_in_middle_rejected(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # A unicode suffix is trailing garbage; reject rather than accept
        # the ASCII prefix (SQLite parse-rejects this shape too).
        assert _parse_savepoint_name("fooé") is None

    def test_trailing_garbage_after_identifier_rejected(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # Extra tokens after a valid identifier mean it is not a clean
        # SAVEPOINT statement.
        assert _parse_savepoint_name("foo extra junk") is None
        assert _parse_savepoint_name("foo()") is None

    def test_trailing_whitespace_and_comment_accepted(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # SQLite tolerates trailing whitespace and comments.
        assert _parse_savepoint_name("foo  ") == "foo"
        assert _parse_savepoint_name("foo /* x */") == "foo"
        assert _parse_savepoint_name("foo -- comment") == "foo"

    def test_ascii_uppercase_lowercased(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("SP") == "sp"

    def test_savepoint_with_leading_digit_does_not_mutate_tracker(
        self, conn: DqliteConnection
    ) -> None:
        # Leading-digit name is parser-rejected, so the tracker stays untouched.
        conn._update_tx_flags_from_sql("SAVEPOINT 1foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False


class TestRollbackTransactionKeyword:
    """The optional TRANSACTION keyword in ROLLBACK [TRANSACTION] [TO
    [SAVEPOINT] name] must be classified correctly in both forms."""

    def test_rollback_transaction_to_savepoint_keeps_outer_tx_open(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["sp"]
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION TO SAVEPOINT sp")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["sp"]

    def test_rollback_transaction_to_savepoint_no_savepoint_keyword(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION TO sp")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["sp"]

    def test_rollback_transaction_full_clears(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION")
        assert conn._in_transaction is False
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False

    def test_rollback_transaction_unwinds_inner_savepoints(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION TO outer")
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True


class TestTxFlagsCommentPrefixed:
    """The tracker must recognise transaction-control statements preceded
    by ``--`` or ``/* */`` comments; otherwise a comment-prefixed BEGIN /
    COMMIT / SAVEPOINT bypasses it and poisons the pool."""

    def test_block_comment_before_savepoint_starts_tx(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("/* annotation */ SAVEPOINT s1")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["s1"]

    def test_line_comment_before_savepoint_starts_tx(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("-- header\nSAVEPOINT s1")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["s1"]

    def test_block_comment_before_begin_sets_in_transaction(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("/* x */ BEGIN")
        assert conn._in_transaction is True

    def test_block_comment_before_commit_clears_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("/* x */ COMMIT")
        assert conn._in_transaction is False

    def test_block_comment_before_release_pops_stack(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT s1")
        conn._update_tx_flags_from_sql("/* x */ RELEASE s1")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_block_comment_before_rollback_to_targets_correct_frame(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("/* x */ ROLLBACK TO outer")
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True

    def test_no_op_when_only_comments(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("-- only a comment")
        conn._update_tx_flags_from_sql("/* only a comment */")
        assert conn._in_transaction is False


class TestSavepointDuplicateNameLIFO:
    """Duplicate savepoint names use the most recently created one (LIFO);
    the tracker must reverse-search the stack to honour this."""

    def test_release_with_duplicate_name_pops_innermost_only(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        assert conn._savepoint_stack == ["sp", "sp"]
        # RELEASE sp removes only the innermost matching frame (and frames
        # above it, of which there are none here).
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == ["sp"]
        assert conn._in_transaction is True

    def test_release_with_duplicate_name_clears_tx_when_outer_released(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        assert conn._savepoint_implicit_begin is True
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == ["sp"]
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_rollback_to_with_duplicate_name_targets_innermost(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        # ROLLBACK TO sp targets the innermost sp (index 1), not the outer
        # one; the matched frame stays, ``inner`` is removed.
        conn._update_tx_flags_from_sql("ROLLBACK TO sp")
        assert conn._savepoint_stack == ["sp", "sp"]
        assert conn._in_transaction is True

    def test_release_with_single_element_stack_clears_stack(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT only")
        assert conn._savepoint_stack == ["only"]
        conn._update_tx_flags_from_sql("RELEASE only")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False

    def test_rollback_to_with_three_duplicate_names_targets_innermost(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        # ROLLBACK TO sp targets the innermost (index 2); matched frame
        # stays, no deeper frames exist, so the stack is unchanged.
        conn._update_tx_flags_from_sql("ROLLBACK TO sp")
        assert conn._savepoint_stack == ["sp", "sp", "sp"]
        assert conn._in_transaction is True

    def test_release_with_mixed_names_and_inner_duplicate_pops_correct_range(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT a")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT b")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        # RELEASE sp targets the innermost (index 3) and pops it + frames
        # above (none), leaving ["a", "sp", "b"].
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == ["a", "sp", "b"]
        assert conn._in_transaction is True


class TestKeywordBoundaryUnderscoreAndAlnum:
    """``_`` is an identifier-continuation char in SQLite but ``str.isalnum``
    rejects it, so a ``not isalnum`` boundary check would mis-split
    ``SAVEPOINT_foo`` into keyword + name. Checked across all six keyword
    sites (BEGIN, SAVEPOINT, RELEASE, ROLLBACK, COMMIT, END)."""

    def test_savepoint_underscore_identifier_not_split(self) -> None:
        """``SAVEPOINT_foo`` is one bareword; the keyword must not be stripped."""
        from dqliteclient.connection import _parse_release_name

        assert _parse_release_name(" SAVEPOINT_foo") == "savepoint_foo"

    def test_release_savepoint_savepoint_underscore_foo_passes_through(self) -> None:
        """``RELEASE SAVEPOINT SAVEPOINT_foo`` releases ``SAVEPOINT_foo``;
        the keyword is consumed once, the trailing bareword is the name."""
        from dqliteclient.connection import _parse_release_name

        assert _parse_release_name(" SAVEPOINT SAVEPOINT_foo") == "savepoint_foo"

    def test_begin_underscore_identifier_does_not_set_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        """``BEGIN_foo`` is a bareword, not the BEGIN keyword."""
        conn._update_tx_flags_from_sql("BEGIN_foo")
        assert conn._in_transaction is False

    def test_savepoint_keyword_followed_by_underscore_treated_as_bareword(
        self, conn: DqliteConnection
    ) -> None:
        """``SAVEPOINT_foo`` (no space) is a bareword; nothing is pushed."""
        conn._update_tx_flags_from_sql("SAVEPOINT_foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False

    def test_release_keyword_followed_by_underscore_treated_as_bareword(
        self, conn: DqliteConnection
    ) -> None:
        """``RELEASE_foo`` is a bareword, not a RELEASE statement."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT _foo")
        conn._update_tx_flags_from_sql("RELEASE_foo")
        assert conn._savepoint_stack == ["_foo"]
        assert conn._in_transaction is True

    def test_commit_underscore_identifier_does_not_close_transaction(
        self, conn: DqliteConnection
    ) -> None:
        """``COMMIT_foo`` is a bareword, not the COMMIT keyword."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("COMMIT_foo")
        assert conn._in_transaction is True

    def test_rollback_transaction_underscore_identifier_treated_as_bareword(
        self, conn: DqliteConnection
    ) -> None:
        """``ROLLBACK TRANSACTION_foo`` cannot strip ``TRANSACTION``; the
        tail has no TO prefix, so the plain-ROLLBACK path runs."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION_foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_rollback_no_tx_defensively_clears_stack(self, conn: DqliteConnection) -> None:
        """A plain ROLLBACK clears all three of stack / implicit-begin /
        untracked-flag so drift state cannot leak past it."""
        # Drift state: stack populated but tx flag false.
        conn._savepoint_stack = ["x"]
        conn._savepoint_implicit_begin = True
        conn._has_untracked_savepoint = True
        conn._in_transaction = False
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False
        assert conn._has_untracked_savepoint is False

    def test_rollback_to_underscore_identifier_not_misclassified(
        self, conn: DqliteConnection
    ) -> None:
        """``ROLLBACK TO_FOO`` has a bareword tail; the TO-prefix check
        respects the keyword boundary and falls into plain-ROLLBACK."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("ROLLBACK TO_FOO")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_rollback_with_tab_separator_recognised_as_savepoint_form(
        self, conn: DqliteConnection
    ) -> None:
        """A tab between keywords is whitespace; ``ROLLBACK\\tTO sp`` is a
        savepoint rollback like the space-separated form."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("ROLLBACK\tTO outer")
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True

    def test_rollback_with_newline_separator_recognised_as_savepoint_form(
        self, conn: DqliteConnection
    ) -> None:
        """Newline between keywords also classifies as savepoint rollback."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("ROLLBACK\nTO outer")
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True

    def test_savepoint_dollar_or_dot_still_split_correctly(self, conn: DqliteConnection) -> None:
        """``$`` and ``.`` are not identifier chars, so the keyword strips;
        the parser then rejects ``$foo`` / ``.foo`` and the tracker is
        untouched."""
        conn._update_tx_flags_from_sql("SAVEPOINT$foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        conn._update_tx_flags_from_sql("SAVEPOINT.foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False


class TestMultiStatementExecTracker:
    """``_update_tx_flags_from_sql`` walks ``;``-separated pieces: the
    server's EXEC path runs every statement, so the tracker must classify
    each piece or desync from server-side state."""

    def test_savepoint_savepoint_pushes_both_frames(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT a; SAVEPOINT b;")
        assert conn._savepoint_stack == ["a", "b"]
        assert conn._in_transaction is True
        assert conn._savepoint_implicit_begin is True

    def test_begin_then_savepoint_in_one_call(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN; SAVEPOINT inner;")
        assert conn._in_transaction is True
        assert conn._savepoint_implicit_begin is False
        assert conn._savepoint_stack == ["inner"]

    def test_savepoint_then_release_clears_correctly(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp; RELEASE sp;")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False

    def test_string_literal_with_semicolon_not_split(self, conn: DqliteConnection) -> None:
        """A ``;`` inside ``'...'`` must NOT split the statement."""
        conn._update_tx_flags_from_sql(
            "INSERT INTO t VALUES ('payload; SAVEPOINT injected'); SAVEPOINT real;"
        )
        assert conn._savepoint_stack == ["real"]

    def test_double_quoted_identifier_with_semicolon_not_split(
        self, conn: DqliteConnection
    ) -> None:
        """A ``;`` inside ``"..."`` must NOT split."""
        conn._update_tx_flags_from_sql('SELECT * FROM "weird;name"; BEGIN;')
        assert conn._in_transaction is True

    def test_block_comment_with_semicolon_not_split(self, conn: DqliteConnection) -> None:
        """A ``;`` inside ``/* ... */`` must NOT split."""
        conn._update_tx_flags_from_sql("/* one ; two */ BEGIN; SAVEPOINT inner;")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["inner"]

    def test_line_comment_with_semicolon_not_split(self, conn: DqliteConnection) -> None:
        """A ``;`` inside ``--`` to end-of-line must NOT split."""
        conn._update_tx_flags_from_sql("BEGIN -- a ; b\n; SAVEPOINT inner;")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["inner"]

    def test_no_semicolon_takes_fast_path(self, conn: DqliteConnection) -> None:
        """Single-statement SQL with no ``;`` takes the fast path unchanged."""
        conn._update_tx_flags_from_sql("SAVEPOINT solo")
        assert conn._savepoint_stack == ["solo"]
        assert conn._in_transaction is True

    def test_trailing_semicolon_only_takes_fast_path(self, conn: DqliteConnection) -> None:
        """A trailing ``;`` produces one piece; same as the no-semicolon form."""
        conn._update_tx_flags_from_sql("SAVEPOINT solo;")
        assert conn._savepoint_stack == ["solo"]


class TestSplitTopLevelStatements:
    """Direct splitter tests for tokenisation edge-cases."""

    def test_doubled_single_quote_inside_string(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        assert _split_top_level_statements("SELECT 'a''b;c'; BEGIN") == [
            "SELECT 'a''b;c'",
            "BEGIN",
        ]

    def test_doubled_double_quote_inside_identifier(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        assert _split_top_level_statements('SELECT "a""b;c"; BEGIN') == [
            'SELECT "a""b;c"',
            "BEGIN",
        ]

    def test_square_bracket_identifier_terminates_only_at_close(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        # Square-bracket identifiers don't escape; the first ``]`` ends them.
        assert _split_top_level_statements("SELECT [a;b]; BEGIN") == [
            "SELECT [a;b]",
            "BEGIN",
        ]

    def test_backtick_identifier_with_doubled_escape(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        assert _split_top_level_statements("SELECT `a``b;c`; BEGIN") == [
            "SELECT `a``b;c`",
            "BEGIN",
        ]

    def test_unterminated_string_literal_eats_to_eof(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        # Malformed input must not crash; the whole tail becomes one piece.
        assert _split_top_level_statements("SELECT 'unterminated; BEGIN") == [
            "SELECT 'unterminated; BEGIN"
        ]

    def test_empty_pieces_dropped(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        assert _split_top_level_statements(";;BEGIN;;") == ["BEGIN"]

    def test_empty_input_returns_empty_list(self) -> None:
        from dqliteclient.connection import _split_top_level_statements

        assert _split_top_level_statements("") == []
        assert _split_top_level_statements("   ") == []
        assert _split_top_level_statements(";;") == []


class TestSavepointEmbeddedComment:
    """Comments are whitespace even between a SAVEPOINT verb and its name,
    so the name parser must strip ``SAVEPOINT /* x */ sp`` to match the
    server (else it falls into the conservative untracked branch)."""

    def test_savepoint_with_embedded_block_comment_pushes_frame(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT /* x */ sp")
        assert conn._savepoint_stack == ["sp"]
        assert conn._in_transaction is True
        assert conn._savepoint_implicit_begin is True

    def test_savepoint_with_embedded_line_comment_pushes_frame(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT -- x\n sp")
        assert conn._savepoint_stack == ["sp"]
        assert conn._in_transaction is True

    def test_release_with_embedded_block_comment_pops_frame(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("RELEASE /* x */ sp")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_release_savepoint_with_embedded_block_comment_pops_frame(
        self, conn: DqliteConnection
    ) -> None:
        """Embedded comment between RELEASE and SAVEPOINT keyword."""
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("RELEASE /* x */ SAVEPOINT sp")
        assert conn._savepoint_stack == []

    def test_rollback_to_with_embedded_block_comment_unwinds(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("ROLLBACK TO /* x */ outer")
        assert conn._savepoint_stack == ["outer"]

    def test_savepoint_with_embedded_comment_does_not_set_untracked_flag(
        self, conn: DqliteConnection
    ) -> None:
        """A comment-prefixed savepoint is tracked, so the untracked flag
        must NOT fire."""
        conn._update_tx_flags_from_sql("SAVEPOINT /* x */ sp")
        assert conn._has_untracked_savepoint is False
