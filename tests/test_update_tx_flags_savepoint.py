"""Pin ``_update_tx_flags_from_sql`` SAVEPOINT / RELEASE / ROLLBACK TO
tracking.

A bare ``SAVEPOINT name`` outside an active transaction triggers the
SQLite implicit BEGIN: the savepoint is the outer frame of an
auto-begun transaction. The corresponding ``RELEASE SAVEPOINT name``
ends that transaction. Without tracking these, ``in_transaction``
lies about server-side state — and ``transaction()`` would raise the
misleading "Nested transactions are not supported" diagnostic when
the user (correctly) tried to wrap a subsequent operation in an
explicit transaction context.

Mirror the stdlib ``sqlite3.Connection`` semantics:

    >>> import sqlite3
    >>> db = sqlite3.connect(":memory:")
    >>> cur = db.cursor()
    >>> cur.execute("SAVEPOINT sp1")
    >>> db.in_transaction
    True

The pin: the prefix-sniff classifier mirrors stdlib for the autobegin
path, while leaving SAVEPOINT-inside-explicit-BEGIN as no-op (the
existing rule, which is correct: SAVEPOINTs nested under a real BEGIN
do not change the outer transaction's boundary).
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
        # SQLite accepts both ``RELEASE name`` and
        # ``RELEASE SAVEPOINT name``; both forms must end the
        # auto-begun transaction.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("RELEASE sp1")
        assert conn._in_transaction is False

    def test_savepoint_inside_explicit_begin_does_not_change_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        # The existing rule (SAVEPOINTs under a real BEGIN do not
        # change the outer boundary) must remain correct.
        conn._update_tx_flags_from_sql("BEGIN")
        assert conn._in_transaction is True
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        assert conn._in_transaction is True
        # RELEASE inside a real BEGIN does not end the transaction.
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT sp1")
        assert conn._in_transaction is True
        # Only COMMIT closes it.
        conn._update_tx_flags_from_sql("COMMIT")
        assert conn._in_transaction is False

    def test_nested_savepoint_release_of_inner_keeps_outer_active(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        assert conn._in_transaction is True
        # RELEASE inner — outer auto-begun savepoint still active.
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT inner")
        assert conn._in_transaction is True
        # RELEASE outer — now the autobegin transaction ends.
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT outer")
        assert conn._in_transaction is False

    def test_rollback_to_savepoint_does_not_change_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        # Per SQLite spec, ROLLBACK TO leaves the named savepoint
        # active — the transaction boundary is unchanged.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("ROLLBACK TO SAVEPOINT sp1")
        assert conn._in_transaction is True

    def test_rollback_to_unwinds_intermediate_frames(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT mid")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        # ROLLBACK TO outer: per SQLite spec, frames above ``outer``
        # (mid, inner) are removed; ``outer`` stays active.
        conn._update_tx_flags_from_sql("ROLLBACK TO SAVEPOINT outer")
        assert conn._in_transaction is True
        # RELEASE outer ends the autobegin transaction.
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT outer")
        assert conn._in_transaction is False

    def test_plain_rollback_clears_savepoint_state(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._in_transaction is False

    def test_release_of_unknown_name_is_a_no_op_on_tracker(self, conn: DqliteConnection) -> None:
        # The server will reject a RELEASE of a name not on the stack,
        # but the helper must not crash (e.g. with IndexError) on the
        # path. The flag stays whatever it was before.
        conn._update_tx_flags_from_sql("SAVEPOINT sp1")
        before = conn._in_transaction
        conn._update_tx_flags_from_sql("RELEASE SAVEPOINT does_not_exist")
        assert conn._in_transaction is before

    def test_savepoint_with_double_quoted_name_not_tracked(self, conn: DqliteConnection) -> None:
        # Double-quoted identifiers are case-sensitive in SQLite, but
        # the unquoted-identifier branch lowercases. Tracking the
        # quoted form would let a later unquoted RELEASE collide
        # against the lowercased entry and pop the local stack while
        # the server (case-sensitive) refuses. Fall back to no-op
        # tracking for quoted names — same precedent as backtick /
        # square-bracket / unicode identifiers.
        conn._update_tx_flags_from_sql('SAVEPOINT "weird name"')
        assert conn._in_transaction is False
        assert conn._savepoint_stack == []

    def test_savepoint_quoted_then_release_unquoted_does_not_pop_stack(
        self, conn: DqliteConnection
    ) -> None:
        # Regression pin for the case-sensitivity divergence: a
        # quoted SAVEPOINT is not tracked, so a subsequent unquoted
        # RELEASE finds nothing on the stack and is a no-op.
        conn._update_tx_flags_from_sql('SAVEPOINT "MyPoint"')
        conn._update_tx_flags_from_sql("RELEASE mypoint")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_tracked_savepoint_after_untracked_does_not_claim_implicit_begin(
        self, conn: DqliteConnection
    ) -> None:
        # The server's autobegin happened on the outer untracked
        # ``"Foo"`` frame, not on the inner tracked ``inner``. The
        # tracker must NOT promote ``inner`` to the autobegin frame:
        # doing so would let a later RELEASE inner flip
        # _in_transaction=False even though the server still holds
        # the autobegun tx (via the outer untracked frame).
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        assert conn._has_untracked_savepoint is True
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        assert conn._savepoint_stack == ["inner"]
        assert conn._savepoint_implicit_begin is False
        # Untracked-savepoint flag survives the inner push.
        assert conn._has_untracked_savepoint is True

    def test_release_inner_tracked_after_untracked_outer_keeps_untracked_flag(
        self, conn: DqliteConnection
    ) -> None:
        # Sequence: untracked outer, tracked inner, RELEASE inner.
        # Pool reset must still fire (untracked-flag still True), and
        # _in_transaction should not be flipped to False purely on
        # the basis of an empty tracked stack — the outer autobegun
        # tx is still server-side alive.
        conn._update_tx_flags_from_sql('SAVEPOINT "Foo"')
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("RELEASE inner")
        assert conn._savepoint_stack == []
        # _savepoint_implicit_begin stays False (was never claimed).
        assert conn._savepoint_implicit_begin is False
        assert conn._has_untracked_savepoint is True

    def test_savepoint_quoted_then_release_quoted_same_text_no_op(
        self, conn: DqliteConnection
    ) -> None:
        # Same as the above: even matching-text quoted RELEASE is
        # not tracked, because the SAVEPOINT itself was not pushed.
        conn._update_tx_flags_from_sql('SAVEPOINT "MyPoint"')
        conn._update_tx_flags_from_sql('RELEASE "MyPoint"')
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_savepoint_name_case_insensitive_match(self, conn: DqliteConnection) -> None:
        # Unquoted SQLite identifiers are case-insensitive.
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
    """Pin the bare-identifier shape contract for ``_parse_savepoint_name``.

    The docstring promises ASCII-only alphanumerics with a leading
    non-digit. SQLite's tokenizer accepts unicode letters (and rejects
    leading-digit names) but Python's ``str.isalnum`` and ``str.lower``
    handle non-ASCII with normalisation rules that may not match the
    server's identifier-fold. Reject up front so the local stack stays
    in step with the server.
    """

    def test_leading_digit_returns_none(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("1foo") is None

    def test_leading_underscore_accepted(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("_sp") == "_sp"

    def test_unicode_letter_returns_none(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("αβγ") is None  # αβγ

    def test_unicode_in_middle_rejected(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # ASCII prefix followed by a unicode suffix is trailing garbage
        # — SQLite would parse-reject this shape too, so the tracker
        # falls through to the parser-rejected branch (None) rather
        # than silently accepting the ASCII prefix.
        assert _parse_savepoint_name("fooé") is None

    def test_trailing_garbage_after_identifier_rejected(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # Forward-defence: extra non-whitespace, non-comment tokens
        # after a valid identifier mean the input is not a clean
        # SAVEPOINT statement.
        assert _parse_savepoint_name("foo extra junk") is None
        assert _parse_savepoint_name("foo()") is None

    def test_trailing_whitespace_and_comment_accepted(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # SQLite tolerates whitespace and comments at the statement
        # tail; the parser should agree.
        assert _parse_savepoint_name("foo  ") == "foo"
        assert _parse_savepoint_name("foo /* x */") == "foo"
        assert _parse_savepoint_name("foo -- comment") == "foo"

    def test_ascii_uppercase_lowercased(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        assert _parse_savepoint_name("SP") == "sp"

    def test_savepoint_with_leading_digit_does_not_mutate_tracker(
        self, conn: DqliteConnection
    ) -> None:
        # End-to-end: the leading-digit name is rejected by the parser,
        # so the tracker stays untouched even if the server were to
        # accept it (it does not — SQLite parse-rejects).
        conn._update_tx_flags_from_sql("SAVEPOINT 1foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False


class TestRollbackTransactionKeyword:
    """Pin SQLite grammar's optional TRANSACTION keyword in ROLLBACK
    statements. Per https://www.sqlite.org/lang_transaction.html the
    rollback grammar is:

        ROLLBACK [TRANSACTION] [TO [SAVEPOINT] name]

    The TRANSACTION keyword is optional in BOTH the full-rollback and
    the rollback-to-savepoint forms. The tracker must classify each
    form correctly or the local stack desyncs from the server."""

    def test_rollback_transaction_to_savepoint_keeps_outer_tx_open(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        assert conn._in_transaction is True
        assert conn._savepoint_stack == ["sp"]
        # ROLLBACK TRANSACTION TO SAVEPOINT sp: rolls back to sp without
        # removing it; outer transaction remains open.
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
        # ROLLBACK TRANSACTION (no TO clause): full rollback.
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION")
        assert conn._in_transaction is False
        assert conn._savepoint_stack == []
        assert conn._savepoint_implicit_begin is False

    def test_rollback_transaction_unwinds_inner_savepoints(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        # ROLLBACK TRANSACTION TO outer: removes inner, keeps outer.
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION TO outer")
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True


class TestTxFlagsCommentPrefixed:
    """Pin the tracker's recognition of transaction-control statements
    preceded by SQL comments. SQLite accepts both ``--`` and ``/* */``
    comments as leading whitespace and runs the post-comment statement.
    Without comment stripping, a comment-prefixed BEGIN / COMMIT /
    SAVEPOINT silently bypasses the tracker — pool poisoning surface."""

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
        # All-comment input is no-op (matches the dbapi cursor behaviour).
        conn._update_tx_flags_from_sql("-- only a comment")
        conn._update_tx_flags_from_sql("/* only a comment */")
        assert conn._in_transaction is False


class TestSavepointDuplicateNameLIFO:
    """Pin SQLite's documented LIFO semantics for duplicate savepoint
    names. Per https://www.sqlite.org/lang_savepoint.html the name of
    a savepoint need not be unique; if multiple savepoints share a
    name, SQLite uses the most recently created one. The tracker must
    reverse-search the stack to honour this contract."""

    def test_release_with_duplicate_name_pops_innermost_only(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        assert conn._savepoint_stack == ["sp", "sp"]
        # RELEASE sp must remove only the innermost matching frame
        # (and any deeper frames, of which there are none here).
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == ["sp"]
        # Outer transaction must still be open.
        assert conn._in_transaction is True

    def test_release_with_duplicate_name_clears_tx_when_outer_released(
        self, conn: DqliteConnection
    ) -> None:
        # Autobegin scenario: SAVEPOINT was the implicit outer frame.
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        assert conn._savepoint_implicit_begin is True
        # First RELEASE pops innermost; outer sp still active.
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == ["sp"]
        assert conn._in_transaction is True
        # Second RELEASE pops the outer; autobegin tx ends.
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
        # ROLLBACK TO sp must target the innermost sp (index 1), not
        # the outer one (index 0). After rollback: ["sp", "sp"] —
        # outer sp still there, inner sp still there (matched-frame
        # not removed by ROLLBACK TO), inner removed.
        conn._update_tx_flags_from_sql("ROLLBACK TO sp")
        assert conn._savepoint_stack == ["sp", "sp"]
        assert conn._in_transaction is True

    def test_release_with_single_element_stack_clears_stack(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT only")
        assert conn._savepoint_stack == ["only"]
        conn._update_tx_flags_from_sql("RELEASE only")
        assert conn._savepoint_stack == []
        # Autobegin path: the implicit tx ends.
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False

    def test_rollback_to_with_three_duplicate_names_targets_innermost(
        self, conn: DqliteConnection
    ) -> None:
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        # ROLLBACK TO sp targets the innermost (index 2). The matched
        # frame stays; deeper frames go (none here). Result: same stack.
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
        # RELEASE sp targets the innermost (index 3) and pops it +
        # all frames above (none above). Result: ["a", "sp", "b"].
        conn._update_tx_flags_from_sql("RELEASE sp")
        assert conn._savepoint_stack == ["a", "sp", "b"]
        assert conn._in_transaction is True


class TestKeywordBoundaryUnderscoreAndAlnum:
    """Pin the keyword/identifier boundary check.

    Python's ``str.isalnum`` returns False for ``_``, but SQLite's
    identifier tokenizer treats ``_`` as a continuation character. A
    boundary check that uses ``not isalnum`` will mis-split identifiers
    like ``SAVEPOINT_foo`` into the keyword ``SAVEPOINT`` followed by
    a separate name ``_foo`` — pushing the wrong name onto the local
    stack while the server (correctly) creates ``SAVEPOINT_foo``.

    Verified across the six keyword sites: BEGIN, SAVEPOINT, RELEASE,
    ROLLBACK [TRANSACTION], COMMIT, END — all share the same boundary
    helper.
    """

    def test_savepoint_underscore_identifier_not_split(self) -> None:
        """``_parse_release_name`` must NOT strip ``SAVEPOINT`` when the
        next character is ``_`` — the whole token is a bareword."""
        from dqliteclient.connection import _parse_release_name

        assert _parse_release_name(" SAVEPOINT_foo") == "savepoint_foo"

    def test_release_savepoint_savepoint_underscore_foo_passes_through(self) -> None:
        """``RELEASE SAVEPOINT SAVEPOINT_foo`` must release the
        identifier ``SAVEPOINT_foo`` — the inner ``SAVEPOINT`` keyword
        is consumed once; the trailing ``SAVEPOINT_foo`` is the name."""
        from dqliteclient.connection import _parse_release_name

        assert _parse_release_name(" SAVEPOINT SAVEPOINT_foo") == "savepoint_foo"

    def test_begin_underscore_identifier_does_not_set_in_transaction(
        self, conn: DqliteConnection
    ) -> None:
        """``BEGIN_foo`` is a bareword (probably not a real DDL but the
        parser must not classify it as the BEGIN keyword)."""
        conn._update_tx_flags_from_sql("BEGIN_foo")
        assert conn._in_transaction is False

    def test_savepoint_keyword_followed_by_underscore_treated_as_bareword(
        self, conn: DqliteConnection
    ) -> None:
        """``SAVEPOINT_foo`` (no space) is a single bareword — the
        prefix-sniff classifier must not push anything onto the local
        savepoint stack."""
        conn._update_tx_flags_from_sql("SAVEPOINT_foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False

    def test_release_keyword_followed_by_underscore_treated_as_bareword(
        self, conn: DqliteConnection
    ) -> None:
        """``RELEASE_foo`` is a single bareword — must not be classified
        as a RELEASE statement."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT _foo")
        # RELEASE_foo (no space) is a bareword; the tracker must not
        # pop the stack.
        conn._update_tx_flags_from_sql("RELEASE_foo")
        assert conn._savepoint_stack == ["_foo"]
        assert conn._in_transaction is True

    def test_commit_underscore_identifier_does_not_close_transaction(
        self, conn: DqliteConnection
    ) -> None:
        """``COMMIT_foo`` is a bareword — must not close the
        transaction the way the COMMIT keyword would."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("COMMIT_foo")
        assert conn._in_transaction is True

    def test_rollback_transaction_underscore_identifier_treated_as_bareword(
        self, conn: DqliteConnection
    ) -> None:
        """The TRANSACTION keyword strip in ROLLBACK must respect the
        underscore boundary too — ``ROLLBACK TRANSACTION_foo`` cannot
        strip ``TRANSACTION``. The whole tail is a bareword that the
        ROLLBACK branch then ignores (no TO prefix), falling into the
        plain-ROLLBACK path."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        # ``TRANSACTION_foo`` is not the TRANSACTION keyword. The tail
        # also lacks a TO prefix, so the plain-ROLLBACK branch runs:
        # the transaction is fully rolled back.
        conn._update_tx_flags_from_sql("ROLLBACK TRANSACTION_foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_rollback_no_tx_defensively_clears_stack(self, conn: DqliteConnection) -> None:
        """The else-branch of the no-tx ROLLBACK clear was previously
        zeroing only ``_has_untracked_savepoint``. Defensive symmetry
        with the if-branch and with close()/_invalidate: clear all
        three of stack / implicit-begin / untracked-flag so a future
        state-machine drift (e.g., a code path that pushes without
        flipping ``_in_transaction``) does not leak stale state past
        a ROLLBACK."""
        # Hypothetical drift state: stack populated but tx flag false.
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
        """``ROLLBACK TO_FOO`` is a bareword tail — the ``TO`` prefix
        check must respect the keyword boundary, otherwise the parser
        would treat ``_FOO`` as a savepoint name and pop frames the
        server never touched. SQLite parse-rejects this shape; the
        tracker should fall through to the plain-ROLLBACK path."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT sp")
        conn._update_tx_flags_from_sql("ROLLBACK TO_FOO")
        # Plain-ROLLBACK semantics: full clear.
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False

    def test_rollback_with_tab_separator_recognised_as_savepoint_form(
        self, conn: DqliteConnection
    ) -> None:
        """SQLite's tokenizer treats tab as whitespace anywhere between
        keywords. ``ROLLBACK\\tTO sp`` must be classified as a savepoint
        rollback (only frames above sp popped, ``_in_transaction``
        unchanged), matching the space-separated form."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("ROLLBACK\tTO outer")
        # ROLLBACK TO popped frames above ``outer``; ``outer`` stays.
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True

    def test_rollback_with_newline_separator_recognised_as_savepoint_form(
        self, conn: DqliteConnection
    ) -> None:
        """Newline between keywords also classifies as savepoint
        rollback, mirroring SQLite's whitespace tolerance."""
        conn._update_tx_flags_from_sql("BEGIN")
        conn._update_tx_flags_from_sql("SAVEPOINT outer")
        conn._update_tx_flags_from_sql("SAVEPOINT inner")
        conn._update_tx_flags_from_sql("ROLLBACK\nTO outer")
        assert conn._savepoint_stack == ["outer"]
        assert conn._in_transaction is True

    def test_savepoint_dollar_or_dot_still_split_correctly(self, conn: DqliteConnection) -> None:
        """Negative pin: ``$`` and ``.`` are NOT identifier characters
        in SQLite, so ``SAVEPOINT$foo`` / ``SAVEPOINT.foo`` are still
        ``SAVEPOINT`` + a non-identifier tail — the keyword-boundary
        check correctly accepts the strip. The parser then rejects
        ``$foo`` / ``.foo`` as invalid bare identifiers, leaving the
        tracker untouched."""
        conn._update_tx_flags_from_sql("SAVEPOINT$foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        conn._update_tx_flags_from_sql("SAVEPOINT.foo")
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False


class TestMultiStatementExecTracker:
    """Pin: ``_update_tx_flags_from_sql`` walks ``;``-separated pieces.

    The dqlite server's EXEC path iterates the statement list, so a
    single ``execute("SAVEPOINT a; SAVEPOINT b;")`` pushes both names
    server-side. Without per-piece classification at the tracker, the
    local stack would push only ``a`` — desyncing from the server.
    """

    def test_savepoint_savepoint_pushes_both_frames(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT a; SAVEPOINT b;")
        assert conn._savepoint_stack == ["a", "b"]
        assert conn._in_transaction is True
        assert conn._savepoint_implicit_begin is True

    def test_begin_then_savepoint_in_one_call(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("BEGIN; SAVEPOINT inner;")
        assert conn._in_transaction is True
        # SAVEPOINT inside an explicit BEGIN does not flip the
        # implicit-begin flag.
        assert conn._savepoint_implicit_begin is False
        assert conn._savepoint_stack == ["inner"]

    def test_savepoint_then_release_clears_correctly(self, conn: DqliteConnection) -> None:
        conn._update_tx_flags_from_sql("SAVEPOINT sp; RELEASE sp;")
        # SAVEPOINT autobegan, RELEASE ended the autobegun tx.
        assert conn._savepoint_stack == []
        assert conn._in_transaction is False
        assert conn._savepoint_implicit_begin is False

    def test_string_literal_with_semicolon_not_split(self, conn: DqliteConnection) -> None:
        """A ``;`` inside ``'...'`` must NOT split the statement —
        otherwise we'd misclassify the string-literal tail as a
        separate statement and drift the tracker."""
        conn._update_tx_flags_from_sql(
            "INSERT INTO t VALUES ('payload; SAVEPOINT injected'); SAVEPOINT real;"
        )
        # Only the trailing real SAVEPOINT must register.
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
        """Single-statement SQL with no ``;`` exercises the existing
        prefix-sniff branch unchanged — pin that the splitter doesn't
        regress the fast path."""
        conn._update_tx_flags_from_sql("SAVEPOINT solo")
        assert conn._savepoint_stack == ["solo"]
        assert conn._in_transaction is True

    def test_trailing_semicolon_only_takes_fast_path(self, conn: DqliteConnection) -> None:
        """A single statement followed by ``;`` produces one piece —
        must classify identically to the no-semicolon form."""
        conn._update_tx_flags_from_sql("SAVEPOINT solo;")
        assert conn._savepoint_stack == ["solo"]


class TestSplitTopLevelStatements:
    """Direct tests for the splitter — easier to pin tokenisation
    edge-cases here than through the tracker."""

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

        # Square-bracket identifiers don't escape — first ``]`` ends.
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

        # Defensive: malformed input must not crash, even if the result
        # is "the whole tail is one statement".
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
    """SQLite tokenizer treats ``/* ... */`` and ``--`` as whitespace
    anywhere in the token stream, including between a SAVEPOINT verb
    and the following name. The tracker's name parser must strip those
    so ``SAVEPOINT /* x */ sp`` is tokenised the same way the server
    will.

    Without the embedded-comment skip, the parser returns None and
    falls into the ``_has_untracked_savepoint`` conservative branch:
    pool reset still fires ROLLBACK so cross-acquirer leaks are
    prevented, but in-task observers see ``in_transaction`` lie
    (False instead of True) and same-task RELEASE on the local-stack
    misses (server has the savepoint; local stack is empty).
    """

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
        """Negative pin: with the parser fix, the comment-prefixed
        savepoint is now properly tracked, so the conservative
        ``_has_untracked_savepoint`` flag should NOT fire."""
        conn._update_tx_flags_from_sql("SAVEPOINT /* x */ sp")
        assert conn._has_untracked_savepoint is False
