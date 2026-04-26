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

    def test_unicode_in_middle_truncates_at_ascii_only(self) -> None:
        from dqliteclient.connection import _parse_savepoint_name

        # ASCII prefix, unicode suffix: parser stops at the boundary
        # rather than swallowing the unicode tail through ``isalnum``.
        assert _parse_savepoint_name("fooé") == "foo"

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
