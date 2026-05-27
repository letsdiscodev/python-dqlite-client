"""Pin: the comment block on ``_initialize_close_unqueued``'s
``except asyncio.CancelledError`` arm accurately describes the
absorb-without-resume semantics rather than promising that the
cancel will resume after the loop.

The method intentionally swallows ``CancelledError`` so the
remaining survivors can be closed; the cancel is NOT re-raised
after the loop. The prior comment said "before letting the cancel
resume" which mis-described the contract. The new comment makes
the absorb-without-resume semantics explicit and points at
``DqliteConnection.__aexit__`` for the analogous idiom.

This is a documentation-accuracy pin so a future contributor
reading the comment cannot make architectural decisions based on
a contract the code does not enforce.
"""

from __future__ import annotations

import inspect

from dqliteclient.pool import ConnectionPool


def test_initialize_close_unqueued_comment_describes_absorb_not_resume() -> None:
    """The cancel-arm comment must NOT say "let the cancel resume"
    (the method never re-raises) and must describe the absorb
    semantics + cross-reference the caller-side bookkeeping."""
    src = inspect.getsource(ConnectionPool._initialize_close_unqueued)

    # The misleading wording must be gone.
    assert "letting the cancel resume" not in src, (
        "comment still promises 'letting the cancel resume' — the code "
        "never re-raises after the loop, so the comment was misleading"
    )
    # The accurate wording must be present.
    assert "intentionally NOT re-raised" in src, (
        "comment must explicitly state the cancel is NOT re-raised"
    )
    # Cross-reference to the analogous absorb idiom.
    assert "DqliteConnection.__aexit__" in src, (
        "comment must cross-reference DqliteConnection.__aexit__ for the analogous absorb idiom"
    )
