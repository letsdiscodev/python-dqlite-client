"""Pin ``_RAFT_BUSY_MESSAGE_FRAGMENTS`` tuple membership.

The tuple at ``connection.py`` is the deliberate extension point for
recognising Raft-side ``SQLITE_BUSY`` wordings that the client must
treat as "the prior write was rejected, the engine-side tx never ran"
— so the tracker's ``_in_transaction`` flag stays untouched. A future
addition (e.g., ``"raft is busy"`` from a server-side rewording) must
be matched by the consumer at the SQLITE_BUSY classification site.

These pins import the tuple by symbol so a refactor that:

* Empties the tuple → fails ``test_tuple_is_non_empty``.
* Drops the canonical fragment → fails the canonical pin.
* Adds a mixed-case fragment → fails the lowercase invariant.

Without these pins, an addition in source could go un-pinned in tests
because every existing test references the bare ``"checkpoint in
progress"`` string directly, not the imported tuple.
"""

from __future__ import annotations

from dqliteclient.connection import _RAFT_BUSY_MESSAGE_FRAGMENTS


def test_canonical_checkpoint_in_progress_present() -> None:
    """The known upstream wording must always be in the set — that's
    the original case the matcher exists for."""
    assert "checkpoint in progress" in _RAFT_BUSY_MESSAGE_FRAGMENTS


def test_tuple_is_non_empty() -> None:
    """An empty tuple would silently disable the entire Raft-busy
    classification path; every SQLITE_BUSY would be treated as
    engine-side and the tracker would clear the tx flag — leaking
    transaction state across legitimate concurrent writers."""
    assert len(_RAFT_BUSY_MESSAGE_FRAGMENTS) >= 1


def test_tuple_entries_are_lowercase() -> None:
    """The matcher lowercases the message text before substring
    search; mixed-case fragments would never match. Pin the
    lowercase invariant so an addition that ignores it is caught at
    test time rather than silently."""
    for fragment in _RAFT_BUSY_MESSAGE_FRAGMENTS:
        assert fragment == fragment.lower(), (
            f"Fragment {fragment!r} contains uppercase; the matcher "
            "lowercases the message text before substring search."
        )


def test_tuple_is_immutable() -> None:
    """Module-level constant is a tuple (not a list / set built on
    every classification). Catches a refactor that converts to a
    mutable container — which would invite per-call mutation bugs
    and undermine the "static extension point" contract."""
    assert isinstance(_RAFT_BUSY_MESSAGE_FRAGMENTS, tuple)


def test_fragments_have_no_leading_or_trailing_whitespace() -> None:
    """A fragment with extra whitespace would silently fail to match
    the typical server message that lacks the same padding. Pin
    cleanliness so an accidental copy-paste with surrounding spaces
    surfaces."""
    for fragment in _RAFT_BUSY_MESSAGE_FRAGMENTS:
        assert fragment == fragment.strip(), (
            f"Fragment {fragment!r} has leading/trailing whitespace; "
            "matcher uses raw substring search so padding would "
            "silently break matching."
        )
