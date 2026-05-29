"""Pin ``_RAFT_BUSY_MESSAGE_FRAGMENTS`` tuple membership and invariants.
The tuple marks Raft-side SQLITE_BUSY wordings as "write rejected, tx never ran"
so the tracker's ``_in_transaction`` flag stays untouched."""

from __future__ import annotations

from dqliteclient.connection import _RAFT_BUSY_MESSAGE_FRAGMENTS


def test_canonical_checkpoint_in_progress_present() -> None:
    """The original upstream wording must always be present."""
    assert "checkpoint in progress" in _RAFT_BUSY_MESSAGE_FRAGMENTS


def test_tuple_is_non_empty() -> None:
    """An empty tuple would disable Raft-busy classification and leak tx state."""
    assert len(_RAFT_BUSY_MESSAGE_FRAGMENTS) >= 1


def test_tuple_entries_are_lowercase() -> None:
    """The matcher lowercases the message first, so mixed-case fragments never match."""
    for fragment in _RAFT_BUSY_MESSAGE_FRAGMENTS:
        assert fragment == fragment.lower(), (
            f"Fragment {fragment!r} contains uppercase; the matcher "
            "lowercases the message text before substring search."
        )


def test_tuple_is_immutable() -> None:
    """The constant must stay a tuple, not a mutable container."""
    assert isinstance(_RAFT_BUSY_MESSAGE_FRAGMENTS, tuple)


def test_fragments_have_no_leading_or_trailing_whitespace() -> None:
    """Padding whitespace would silently break the raw substring match."""
    for fragment in _RAFT_BUSY_MESSAGE_FRAGMENTS:
        assert fragment == fragment.strip(), (
            f"Fragment {fragment!r} has leading/trailing whitespace; "
            "matcher uses raw substring search so padding would "
            "silently break matching."
        )
