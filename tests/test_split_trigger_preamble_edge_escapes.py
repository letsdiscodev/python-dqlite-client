"""Pin: top-level statement splitter handles two trigger-preamble edge
cases that prior tests did not cover.

1. A doubled single-quote (``''``) escape INSIDE a single-quoted
   preamble literal — the ``BEGIN`` / ``;`` inside such a literal must
   not be mistaken for the trigger body start or a statement separator.
2. A non-``TRIGGER`` ``CREATE`` (``CREATE TABLE`` / ``INDEX`` / ``VIEW``)
   must bail out of trigger-body scanning and split as ordinary
   ``;``-separated statements.
"""

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
