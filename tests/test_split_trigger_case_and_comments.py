"""Trigger splitter: ``CASE..END`` inside a body must not decrement trigger_depth
(it ends the CASE, not the block), and comments between CREATE and TRIGGER are
skipped."""

from __future__ import annotations

import pytest

from dqliteclient.connection import _split_top_level_statements


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
