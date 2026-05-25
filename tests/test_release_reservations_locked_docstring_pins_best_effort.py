"""Pin: ``_release_reservations_locked``'s docstring + AssertionError
wording reflect the best-effort semantics of ``asyncio.Lock.locked()``.

``asyncio.Lock.locked()`` reports whether ANY task holds the lock,
not whether the current caller does. The precondition check therefore
catches the "no task holds the lock" misuse but cannot detect a
"different task holds the lock, current task does not" cross-task
violation. The docstring previously overstated the guarantee — readers
trusting the strong wording could build patterns that violate the
actual ownership contract. The wording now flags the limitation
explicitly so the gap is visible at the API boundary.
"""

from __future__ import annotations

from dqliteclient.pool import ConnectionPool


def test_docstring_flags_best_effort_semantics() -> None:
    doc = ConnectionPool._release_reservations_locked.__doc__ or ""
    # The corrected docstring names the limitation in plain terms.
    assert "best-effort" in doc.lower(), (
        "_release_reservations_locked docstring must flag the lock check "
        "as best-effort so readers know cross-task ownership is not enforced"
    )
    assert (
        "no owner tracking" in doc.lower()
        or "owner-tracking" in doc.lower()
        or ("any task" in doc.lower())
    ), (
        "docstring must reference the asyncio.Lock owner-tracking limitation "
        "so the contract is unambiguous"
    )


def test_assertion_error_wording_explains_limitation() -> None:
    """The runtime AssertionError text names the cross-task limitation
    so an operator who trips the check sees the precondition contract
    inline, not just a generic message.
    """
    import inspect

    src = inspect.getsource(ConnectionPool._release_reservations_locked)
    assert "no owner tracking" in src or "cross-task" in src, (
        "AssertionError text must reference the cross-task / no-owner-tracking "
        "limitation so the operator-facing diagnostic is accurate"
    )
