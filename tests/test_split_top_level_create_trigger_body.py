"""``_split_top_level_statements`` keeps a CREATE TRIGGER BEGIN..END body as one
piece: ``;`` inside it is an inner terminator, not an outer statement boundary.
A bare top-level ``BEGIN`` is transaction-control and must still split."""

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
