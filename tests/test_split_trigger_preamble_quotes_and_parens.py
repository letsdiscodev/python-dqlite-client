r"""``_scan_for_trigger_begin`` tracks paren depth and quote-styled identifiers in
the trigger preamble: a ``WHEN (...BEGIN...)`` clause or a quoted ``"BEGIN"``
must not false-match the standalone body-opening ``BEGIN``."""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


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
