"""Trigger splitter keeps compound bodies with nested ``BEGIN``...``END`` blocks
as one piece (savepoint-emulation patterns produce these; inner END must not
close the trigger)."""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


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
