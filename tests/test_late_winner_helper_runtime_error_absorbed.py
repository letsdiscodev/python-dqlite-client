"""Pin: the late-winner helper's RuntimeError is absorbed in
``ConnectionPool.acquire``'s except-arm cleanup so it does not
supplant the user's original cancel exception.

Pre-fix, ``await self._put_back_or_release_late_winner(...)`` ran
under the except-arm without a narrow RuntimeError absorption. A
racing ``engine.dispose()`` closing the loop mid-cleanup caused the
helper's ``_close_best_effort`` / ``_release_reservation`` to raise
``RuntimeError("Event loop is closed")``, and the bare ``raise`` at
the cleanup tail re-raised the helper's exception instead of the
caller's cancel. Forensic operators triaging the cancel-cascade saw
``RuntimeError`` at ``e.__class__`` with the cancel preserved only
on ``__context__``.

Mirrors the sibling discipline at ``pool.acquire``'s shielded close
cleanup which already absorbs ``RuntimeError`` for the same reason.
"""

from __future__ import annotations

import inspect

from dqliteclient import pool as pool_mod


def test_late_winner_helper_calls_are_runtime_error_protected() -> None:
    """Source-level pin: every ``await
    self._put_back_or_release_late_winner(...)`` call site in
    ``ConnectionPool.acquire`` is wrapped in
    ``try: ... except RuntimeError: logger.debug(...)`` so a helper
    ``RuntimeError`` does not supplant the user's cancel.
    """
    src = inspect.getsource(pool_mod.ConnectionPool.acquire)
    # Strip whitespace-only and comment-only lines so a leftover
    # rationale comment doesn't false-satisfy the pin.
    lines = [line for line in src.splitlines() if line.strip()]
    # Find every helper-call line and assert it is preceded by a
    # ``try:`` on the immediately-previous non-empty line. Pre-fix
    # the helper call was raw; post-fix the line is preceded by
    # ``try:`` followed by an indented ``await``.
    helper_call_indices = [
        i for i, line in enumerate(lines) if "await self._put_back_or_release_late_winner(" in line
    ]
    assert helper_call_indices, "expected at least one late-winner helper call site"
    for idx in helper_call_indices:
        # The wrap pattern places ``try:`` on the immediately
        # previous line. Skip leading whitespace-only lines and
        # comments when checking.
        prev = lines[idx - 1].strip()
        # Skip past any contiguous comment lines.
        scan = idx - 1
        while prev.startswith("#") and scan > 0:
            scan -= 1
            prev = lines[scan].strip()
        assert prev == "try:", (
            f"late-winner helper call at line {idx} of acquire() must be "
            f"wrapped in try/except RuntimeError so a helper RuntimeError "
            f"does not supplant the user's original cancel; previous "
            f"non-comment line is {prev!r}"
        )
    # Verify the catch arm exists somewhere in the body. The exact
    # logger string differs between sites but the discipline shape is
    # constant.
    body = "\n".join(lines)
    assert body.count("except RuntimeError:") >= len(helper_call_indices), (
        "every late-winner helper call site must have a paired "
        "`except RuntimeError:` arm; counts mismatch"
    )
