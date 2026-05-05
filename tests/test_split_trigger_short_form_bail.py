"""Pin: ``_scan_for_trigger_begin`` bails out for SQLite's
short-form ``CREATE TRIGGER ... INSERT ...;`` (no BEGIN..END).

A regression that fails to bail (returns the BEGIN-found path
> 0) would treat the rest of the script as trigger body and
silently swallow subsequent statements — subtle wrong-
classification surfacing only on migrations using the short-form
trigger.

Companion to ``test_split_trigger_preamble_quotes_and_parens.py``
which exercises the preamble parser; this pin covers the
``;``-before-BEGIN bail at connection.py:443-444.
"""

from dqliteclient.connection import _scan_for_trigger_begin


def test_short_form_trigger_no_begin_returns_zero() -> None:
    """SQLite short-form trigger has no BEGIN..END; the scanner
    must return 0 on the ``;`` to let the outer splitter treat
    the semicolon as a real statement boundary."""
    sql = "CREATE TRIGGER x AFTER INSERT ON t INSERT INTO log VALUES(1);"
    # `after_create` is just past ``CREATE``.
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result == 0, (
        "short-form trigger (no BEGIN..END) must bail with 0 so "
        "the outer splitter treats the trailing ';' as a real "
        "statement boundary"
    )


def test_full_form_trigger_with_begin_returns_offset() -> None:
    """Positive control: full-form trigger with BEGIN returns the
    offset just past BEGIN so the outer splitter knows to ignore
    semicolons until matching END."""
    sql = "CREATE TRIGGER x AFTER INSERT ON t WHEN (a > 0) BEGIN INSERT INTO log VALUES(1); END;"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result > 0, "full-form trigger must locate BEGIN"
    # Result points just past 'BEGIN'.
    assert sql[result - 5 : result] == "BEGIN"


def test_short_form_with_paren_when_clause_then_no_begin_bails() -> None:
    """A WHEN clause with parens then a short-form body (no BEGIN)
    still bails on the trailing ``;``. Without the bail, the
    paren-balanced scanner would walk past the close-paren and
    treat the post-WHEN body as trigger-body content."""
    sql = "CREATE TRIGGER x AFTER UPDATE OF c ON t WHEN (a > 0) UPDATE other SET x=1;"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result == 0
