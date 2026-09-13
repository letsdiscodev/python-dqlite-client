"""Top-level SQL statement splitting, including CREATE TRIGGER bodies and preamble edge cases."""

from __future__ import annotations

import pytest

from dqliteclient.sql import _scan_for_trigger_begin
from dqliteclient.sql import split_statements as _split_top_level_statements

# ``_split_top_level_statements`` keeps a CREATE TRIGGER BEGIN..END body as one
# piece: ``;`` inside it is an inner terminator, not an outer statement boundary.
# A bare top-level ``BEGIN`` is transaction-control and must still split.


class TestCreateTriggerBodyKeptTogether:
    def test_basic_trigger_body(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER INSERT ON x BEGIN\n"
            "  UPDATE y SET v=1 WHERE id=NEW.id;\n"
            "  DELETE FROM z WHERE id=NEW.id;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1
        assert pieces[0].upper().startswith("CREATE TRIGGER")
        assert pieces[0].rstrip().upper().endswith("END")

    def test_temp_trigger(self) -> None:
        sql = "CREATE TEMP TRIGGER t AFTER INSERT ON x BEGIN\n  UPDATE y SET v=1;\nEND;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_temporary_trigger(self) -> None:
        sql = "CREATE TEMPORARY TRIGGER t AFTER INSERT ON x BEGIN\n  UPDATE y SET v=1;\nEND;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_instead_of_trigger(self) -> None:
        sql = (
            "CREATE TRIGGER t INSTEAD OF INSERT ON v BEGIN\n"
            "  INSERT INTO base VALUES (NEW.x);\n"
            "  UPDATE other SET y=1;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_trigger_followed_by_other_statement(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER INSERT ON x BEGIN\n"
            "  UPDATE y SET v=1;\n"
            "END;\n"
            "INSERT INTO log VALUES (1)"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2
        assert pieces[0].upper().startswith("CREATE TRIGGER")
        assert pieces[1].upper().startswith("INSERT")


class TestRegularBeginEndStillSplits:
    def test_bare_begin_commit_still_split(self) -> None:
        """Top-level ``BEGIN`` is transaction-control, not a trigger body, so
        ``BEGIN; INSERT; COMMIT;`` must still split."""
        sql = "BEGIN; INSERT INTO t VALUES (1); COMMIT;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 3
        assert pieces[0].upper() == "BEGIN"
        assert pieces[1].upper().startswith("INSERT")
        assert pieces[2].upper() == "COMMIT"

    def test_begin_transaction_still_splits(self) -> None:
        sql = "BEGIN TRANSACTION; SELECT 1; COMMIT;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 3

    def test_begin_immediate_still_splits(self) -> None:
        sql = "BEGIN IMMEDIATE; INSERT INTO t VALUES (1); COMMIT;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 3


class TestTriggerKeywordsInsideQuotedContextDoNotFlipMode:
    def test_string_literal_with_create_trigger_text_does_not_enter_trigger_mode(self) -> None:
        sql = "INSERT INTO t VALUES ('CREATE TRIGGER a AFTER INSERT BEGIN'); SELECT 1"
        pieces = _split_top_level_statements(sql)
        # String literal contents are not parsed: splitter sees INSERT then SELECT.
        assert len(pieces) == 2
        assert pieces[0].upper().startswith("INSERT")
        assert pieces[1].upper().startswith("SELECT")

    def test_comment_with_create_trigger_text_does_not_enter_trigger_mode(self) -> None:
        sql = "/* CREATE TRIGGER a AFTER INSERT BEGIN */ SELECT 1; SELECT 2"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2

    def test_keyword_substring_does_not_match(self) -> None:
        """A name that merely STARTS with ``trigger`` must not flip mode."""
        sql = "INSERT INTO triggers VALUES (1); SELECT 1"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2


# ``_split_top_level_statements`` returns all-comment input as a single
# (non-empty) piece; downstream classifiers strip it to a no-op.


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


# Trigger splitter: ``CASE..END`` inside a body must not decrement trigger_depth
# (it ends the CASE, not the block), and comments between CREATE and TRIGGER are
# skipped.


class TestTriggerBodyCaseExpression:
    def test_simple_case_in_update_kept_together(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER UPDATE ON t BEGIN\n"
            "  UPDATE t SET x = CASE WHEN y > 0 THEN 1 ELSE 2 END;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_nested_case(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER UPDATE ON t BEGIN\n"
            "  UPDATE t SET x = CASE WHEN y > 0 THEN "
            "(CASE WHEN z THEN 'a' ELSE 'b' END) ELSE 'c' END;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_multiple_cases_in_body(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER UPDATE ON t BEGIN\n"
            "  UPDATE t SET x = CASE WHEN a THEN 1 END;\n"
            "  UPDATE t SET y = CASE WHEN b THEN 2 ELSE 3 END;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1


class TestTriggerPreambleComments:
    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE/* migration v3 */TRIGGER aud AFTER INSERT ON x BEGIN UPDATE y SET v=1; END;",
            "CREATE -- migration v3\nTRIGGER aud AFTER INSERT ON x BEGIN UPDATE y SET v=1; END;",
            "CREATE TEMP /* tmp */TRIGGER aud AFTER INSERT ON x BEGIN UPDATE y SET v=1; END;",
            "CREATE/*A*/TEMPORARY/*B*/TRIGGER aud AFTER INSERT ON x BEGIN UPDATE y SET v=1; END;",
            "CREATE  -- explanation\n   TRIGGER aud AFTER INSERT ON x BEGIN UPDATE y SET v=1; END;",
        ],
    )
    def test_trigger_preamble_with_comments_kept_together(self, sql: str) -> None:
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1, f"got {len(pieces)} pieces: {pieces!r}"


class TestBasicTriggerBodyRegression:
    def test_basic_trigger_body_kept_together(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER INSERT ON x BEGIN\n"
            "  UPDATE y SET v=1 WHERE id=NEW.id;\n"
            "  DELETE FROM z WHERE id=NEW.id;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_bare_begin_commit_still_split(self) -> None:
        sql = "BEGIN; INSERT INTO t VALUES (1); COMMIT;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 3


# Trigger splitter keeps compound bodies with nested ``BEGIN``...``END`` blocks
# as one piece (savepoint-emulation patterns produce these; inner END must not
# close the trigger).


class TestTriggerBodyNestedBeginEnd:
    def test_compound_trigger_with_nested_begin_end_kept_together(self) -> None:
        sql = (
            "CREATE TRIGGER t AFTER INSERT ON tab BEGIN "
            "  BEGIN "
            "    INSERT INTO other VALUES (NEW.id); "
            "  END; "
            "END; "
            "SELECT 2;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2, (
            f"compound trigger body must remain a single piece; got "
            f"{len(pieces)} pieces: {pieces!r}"
        )
        assert "BEGIN" in pieces[0].upper()
        assert pieces[0].upper().count("END") == 2
        assert pieces[1].strip().upper().startswith("SELECT")

    def test_triply_nested_begin_end_kept_together(self) -> None:
        sql = (
            "CREATE TRIGGER deep AFTER INSERT ON tab BEGIN "
            "  BEGIN "
            "    BEGIN "
            "      INSERT INTO a VALUES (1); "
            "    END; "
            "  END; "
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1
        assert pieces[0].upper().count("END") == 3

    def test_trigger_body_nested_begin_end_with_case_inside(self) -> None:
        """Nested ``BEGIN``/``END`` and a ``CASE`` in one body: the two depth
        counters must be tracked independently."""
        sql = (
            "CREATE TRIGGER mix AFTER UPDATE ON tab BEGIN "
            "  BEGIN "
            "    UPDATE tab SET x = CASE WHEN y > 0 THEN 1 ELSE 2 END; "
            "  END; "
            "END; "
            "SELECT 9;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2
        # Three END tokens: CASE, inner BEGIN, trigger BEGIN.
        assert pieces[0].upper().count("END") == 3
        assert pieces[1].strip().upper().startswith("SELECT")

    def test_nested_begin_end_followed_by_top_level_statements(self) -> None:
        sql = (
            "CREATE TRIGGER t AFTER INSERT ON tab BEGIN "
            "  BEGIN "
            "    INSERT INTO log VALUES (NEW.id); "
            "  END; "
            "END; "
            "INSERT INTO tab VALUES (1); "
            "SELECT * FROM tab;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 3
        assert "CREATE TRIGGER" in pieces[0].upper()
        assert pieces[1].strip().upper().startswith("INSERT")
        assert pieces[2].strip().upper().startswith("SELECT")


# Trigger-preamble edge cases: ``BEGIN``/``;`` inside a doubled-quote (``''``)
# escaped literal is not a body start/separator, and a non-TRIGGER CREATE bails
# out to ordinary ``;``-splitting.


def test_trigger_preamble_doubled_single_quote_escape_keeps_body_together() -> None:
    sql = (
        "CREATE TRIGGER t AFTER UPDATE ON x "
        "WHEN (NEW.a = 'it''s a BEGIN; trap') BEGIN\n"
        " UPDATE y SET v=1;\n END; SELECT 2"
    )
    pieces = _split_top_level_statements(sql)
    assert len(pieces) == 2
    assert pieces[0].upper().startswith("CREATE TRIGGER")
    assert pieces[1].strip() == "SELECT 2"


def test_non_trigger_create_statements_split_normally() -> None:
    pieces = _split_top_level_statements("CREATE TABLE t (a INT); CREATE INDEX i ON t(a); SELECT 1")
    assert len(pieces) == 3


# ``_scan_for_trigger_begin`` tracks paren depth and quote-styled identifiers in
# the trigger preamble: a ``WHEN (...BEGIN...)`` clause or a quoted ``"BEGIN"``
# must not false-match the standalone body-opening ``BEGIN``.


class TestPreambleParenTracking:
    def test_when_clause_with_begin_token_in_paren_does_not_split(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER UPDATE ON x "
            "FOR EACH ROW WHEN (NEW.a = 'BEGIN' OR OLD.b > 0) BEGIN\n"
            "  UPDATE y SET v=1;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1, f"got {len(pieces)} pieces: {pieces!r}"

    def test_nested_when_clauses(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER UPDATE ON x "
            "FOR EACH ROW WHEN ((NEW.a > 0) AND (NEW.b < 10)) BEGIN\n"
            "  UPDATE y SET v=1;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1


class TestPreambleQuotedIdentifiers:
    def test_double_quoted_table_name_in_preamble(self) -> None:
        sql = 'CREATE TRIGGER aud AFTER INSERT ON "my table" BEGIN\n  UPDATE y SET v=1;\nEND;'
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1, f"got {len(pieces)} pieces: {pieces!r}"

    def test_double_quoted_with_doubled_quote_escape(self) -> None:
        # SQLite identifier quoting: "" inside "..." is a literal "
        sql = 'CREATE TRIGGER aud AFTER INSERT ON "weird""name" BEGIN\n  UPDATE y SET v=1;\nEND;'
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_bracketed_table_name_in_preamble(self) -> None:
        sql = "CREATE TRIGGER aud AFTER INSERT ON [my table] BEGIN\n  UPDATE y SET v=1;\nEND;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_backtick_table_name_in_preamble(self) -> None:
        sql = "CREATE TRIGGER aud AFTER INSERT ON `my table` BEGIN\n  UPDATE y SET v=1;\nEND;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_backtick_with_doubled_backtick_escape(self) -> None:
        sql = "CREATE TRIGGER aud AFTER INSERT ON `weird``name` BEGIN\n  UPDATE y SET v=1;\nEND;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1


class TestPreambleComments:
    def test_line_comment_between_trigger_and_begin(self) -> None:
        sql = (
            "CREATE TRIGGER aud AFTER INSERT ON x -- migration v3\nBEGIN\n  UPDATE y SET v=1;\nEND;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_block_comment_between_trigger_and_begin(self) -> None:
        sql = "CREATE TRIGGER aud AFTER INSERT ON x /* notes */ BEGIN\n  UPDATE y SET v=1;\nEND;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1


# ``_scan_for_trigger_begin`` bails (returns 0) on SQLite's short-form
# ``CREATE TRIGGER ... INSERT ...;`` with no BEGIN..END; otherwise it would treat
# the rest of the script as trigger body and swallow subsequent statements.


def test_short_form_trigger_no_begin_returns_zero() -> None:
    sql = "CREATE TRIGGER x AFTER INSERT ON t INSERT INTO log VALUES(1);"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result == 0, (
        "short-form trigger (no BEGIN..END) must bail with 0 so "
        "the outer splitter treats the trailing ';' as a real "
        "statement boundary"
    )


def test_full_form_trigger_with_begin_returns_offset() -> None:
    """Full-form trigger returns the offset just past BEGIN."""
    sql = "CREATE TRIGGER x AFTER INSERT ON t WHEN (a > 0) BEGIN INSERT INTO log VALUES(1); END;"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result > 0, "full-form trigger must locate BEGIN"
    assert sql[result - 5 : result] == "BEGIN"


def test_short_form_with_paren_when_clause_then_no_begin_bails() -> None:
    """A parenthesised WHEN clause then a short-form body (no BEGIN) still bails
    on the trailing ``;`` rather than treating the post-WHEN body as trigger
    content."""
    sql = "CREATE TRIGGER x AFTER UPDATE OF c ON t WHEN (a > 0) UPDATE other SET x=1;"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result == 0
