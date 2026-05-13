"""Pin: trigger splitter handles compound trigger bodies with nested
``BEGIN``...``END`` blocks.

The top-level-statement splitter tracks ``trigger_depth`` so a
compound trigger (``CREATE TRIGGER ... BEGIN ... BEGIN ... END;
END;``) closes correctly. The CASE-expression sibling is already
covered by ``test_split_trigger_case_and_comments.py``; this file
pins the nested-BEGIN/END arm at ``connection.py:298-313``.

Without a pin, a regression that dropped the ``trigger_depth`` ++/--
asymmetry (e.g. a future cleanup arguing 'SQLite triggers can't have
nested BEGIN' — false, savepoint-emulation patterns produce them)
would silently split the trigger body mid-statement and dispatch the
trailing ``END;`` as a top-level statement.
"""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


class TestTriggerBodyNestedBeginEnd:
    def test_compound_trigger_with_nested_begin_end_kept_together(self) -> None:
        """A trigger body that nests ``BEGIN``...``END`` blocks must
        remain ONE logical piece; the inner ``END`` must not close the
        trigger body."""
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
        # The trigger body piece must contain BOTH END tokens.
        assert "BEGIN" in pieces[0].upper()
        assert pieces[0].upper().count("END") == 2
        # The trailing top-level SELECT is the second piece.
        assert pieces[1].strip().upper().startswith("SELECT")

    def test_triply_nested_begin_end_kept_together(self) -> None:
        """Three levels of nesting — trigger_depth must increment AND
        decrement symmetrically. Asymmetric tracking would silently
        corrupt the split here."""
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
        """Combined: nested ``BEGIN``/``END`` AND a ``CASE`` expression
        in the same body. Both depth counters must be tracked
        independently."""
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
        # The trigger piece carries: trigger BEGIN, inner BEGIN, the
        # CASE END, the inner END, and the trigger END — three END
        # tokens (CASE, inner BEGIN, trigger BEGIN).
        assert pieces[0].upper().count("END") == 3
        assert pieces[1].strip().upper().startswith("SELECT")

    def test_nested_begin_end_followed_by_top_level_statements(self) -> None:
        """A compound trigger followed by multiple top-level statements
        — ensures the trigger body closes cleanly and the subsequent
        statements split correctly."""
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
