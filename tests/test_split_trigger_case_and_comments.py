"""Pin: trigger splitter handles CASE...END expressions and comments
between CREATE and TRIGGER.

Two cycle-19 follow-ups:

1. ``CASE WHEN ... END`` inside a trigger body must NOT decrement
   ``trigger_depth`` (the END terminates the CASE expression, not
   the BEGIN..END block).
2. Comments (``--`` line, ``/* */`` block) between ``CREATE`` and
   ``TRIGGER`` (or between ``TEMP`` / ``TEMPORARY`` and ``TRIGGER``)
   must be skipped — migration-tool output routinely interleaves
   metadata comments at this position.
"""

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


class TestRegressionsFromCycle19StillPass:
    def test_basic_trigger_body_kept_together(self) -> None:
        """Sanity: cycle 19's basic regression test still passes."""
        sql = (
            "CREATE TRIGGER aud AFTER INSERT ON x BEGIN\n"
            "  UPDATE y SET v=1 WHERE id=NEW.id;\n"
            "  DELETE FROM z WHERE id=NEW.id;\n"
            "END;"
        )
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 1

    def test_bare_begin_commit_still_split(self) -> None:
        """Sanity: ordinary multi-statement batches still split."""
        sql = "BEGIN; INSERT INTO t VALUES (1); COMMIT;"
        pieces = _split_top_level_statements(sql)
        assert len(pieces) == 3
