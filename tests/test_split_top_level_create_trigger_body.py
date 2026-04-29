"""Pin: ``_split_top_level_statements`` keeps a CREATE TRIGGER body
together as one statement.

SQLite's parser treats ``;`` inside a ``BEGIN..END`` trigger body
as inner-statement terminators that do NOT close the outer DDL
(``parse.y::trigger_cmd_list``). The splitter previously had no
awareness of trigger-body block scope, so a trigger DDL with
embedded semicolons was split into multiple pieces — and the bare
``END`` token at the end then matched the COMMIT/END branch in
``_update_tx_flags_from_sql``, corrupting the transaction tracker.

Scope of the fix is restricted to triggers: a bare ``BEGIN`` is
transaction-control (``BEGIN``, ``BEGIN TRANSACTION``,
``BEGIN DEFERRED/IMMEDIATE/EXCLUSIVE``) and must still split.
Trigger-body mode is entered only after the lexer recognises
``CREATE [TEMP|TEMPORARY] TRIGGER ... BEGIN``.
"""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


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
        """The next statement after the trigger body's terminating ``;``
        must split off correctly."""
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
        """Top-level ``BEGIN`` (transaction-control, NOT trigger body)
        must still split on ``;`` boundaries — otherwise multi-
        statement batches like ``BEGIN; INSERT; COMMIT;`` would be
        glued together."""
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
        # The string literal contents are NOT parsed; the splitter
        # sees an INSERT and a SELECT.
        assert len(pieces) == 2
        assert pieces[0].upper().startswith("INSERT")
        assert pieces[1].upper().startswith("SELECT")

    def test_comment_with_create_trigger_text_does_not_enter_trigger_mode(self) -> None:
        sql = "/* CREATE TRIGGER a AFTER INSERT BEGIN */ SELECT 1; SELECT 2"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2

    def test_keyword_substring_does_not_match(self) -> None:
        """A column or table whose name STARTS with ``trigger`` must
        not flip mode."""
        sql = "INSERT INTO triggers VALUES (1); SELECT 1"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 2
