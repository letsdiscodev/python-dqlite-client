"""Trigger-preamble edge cases: ``BEGIN``/``;`` inside a doubled-quote (``''``)
escaped literal is not a body start/separator, and a non-TRIGGER CREATE bails
out to ordinary ``;``-splitting."""

from __future__ import annotations

from dqliteclient.connection import _split_top_level_statements


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
