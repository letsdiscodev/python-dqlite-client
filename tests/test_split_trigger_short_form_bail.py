"""``_scan_for_trigger_begin`` bails (returns 0) on SQLite's short-form
``CREATE TRIGGER ... INSERT ...;`` with no BEGIN..END; otherwise it would treat
the rest of the script as trigger body and swallow subsequent statements."""

from dqliteclient.connection import _scan_for_trigger_begin


def test_short_form_trigger_no_begin_returns_zero() -> None:
    sql = "CREATE TRIGGER x AFTER INSERT ON t INSERT INTO log VALUES(1);"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result == 0, (
        "short-form trigger (no BEGIN..END) must bail with 0 so "
        "the outer splitter treats the trailing ';' as a real "
        "statement boundary"
    )


def test_full_form_trigger_with_begin_returns_offset() -> None:
    """Full-form trigger returns the offset just past BEGIN."""
    sql = "CREATE TRIGGER x AFTER INSERT ON t WHEN (a > 0) BEGIN INSERT INTO log VALUES(1); END;"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result > 0, "full-form trigger must locate BEGIN"
    assert sql[result - 5 : result] == "BEGIN"


def test_short_form_with_paren_when_clause_then_no_begin_bails() -> None:
    """A parenthesised WHEN clause then a short-form body (no BEGIN) still bails
    on the trailing ``;`` rather than treating the post-WHEN body as trigger
    content."""
    sql = "CREATE TRIGGER x AFTER UPDATE OF c ON t WHEN (a > 0) UPDATE other SET x=1;"
    after_create = len("CREATE")
    result = _scan_for_trigger_begin(sql, after_create, len(sql))
    assert result == 0
