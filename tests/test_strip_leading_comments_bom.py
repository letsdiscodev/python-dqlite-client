"""Pin: client-layer ``_strip_leading_comments`` strips leading
UTF-8 BOM. See dbapi-side test for full rationale.
"""

from __future__ import annotations

from dqliteclient.connection import _starts_with_tx_verb, _strip_leading_comments

_BOM = "﻿"


def test_strip_leading_comments_strips_lone_bom() -> None:
    assert _strip_leading_comments(f"{_BOM}BEGIN") == "BEGIN"


def test_strip_leading_comments_strips_bom_then_whitespace() -> None:
    assert _strip_leading_comments(f"{_BOM}   COMMIT") == "COMMIT"


def test_strip_leading_comments_strips_bom_then_line_comment() -> None:
    assert _strip_leading_comments(f"{_BOM}-- hi\nROLLBACK") == "ROLLBACK"


def test_strip_leading_comments_strips_bom_then_block_comment() -> None:
    assert _strip_leading_comments(f"{_BOM}/* hi */ SAVEPOINT a") == "SAVEPOINT a"


def test_starts_with_tx_verb_recognises_bom_prefixed_begin() -> None:
    assert _starts_with_tx_verb(f"{_BOM}BEGIN") is True


def test_starts_with_tx_verb_recognises_bom_prefixed_commit() -> None:
    assert _starts_with_tx_verb(f"{_BOM}COMMIT") is True
