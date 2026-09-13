"""Client-side argument validation: params, timeouts, bool rejection, max_attempts parity."""

from __future__ import annotations

import pytest

from dqliteclient import validate_timeout
from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


def _make_conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestClientValidateParamsRichRejections:
    def test_rejects_bytearray(self) -> None:
        with pytest.raises(DataError, match="bytearray"):
            _make_conn()._validate_params(bytearray(b"abc"))

    def test_rejects_memoryview(self) -> None:
        with pytest.raises(DataError, match="memoryview"):
            _make_conn()._validate_params(memoryview(b"abc"))

    def test_rejects_dict(self) -> None:
        with pytest.raises(DataError, match="mapping"):
            _make_conn()._validate_params({"a": 1})

    def test_rejects_set(self) -> None:
        with pytest.raises(DataError, match="set"):
            _make_conn()._validate_params({1, 2, 3})

    def test_rejects_frozenset(self) -> None:
        with pytest.raises(DataError, match="set"):
            _make_conn()._validate_params(frozenset({1, 2, 3}))

    def test_str_and_bytes_still_rejected(self) -> None:
        with pytest.raises(DataError, match="str"):
            _make_conn()._validate_params("abc")
        with pytest.raises(DataError, match="bytes"):
            _make_conn()._validate_params(b"abc")

    def test_list_and_tuple_accepted(self) -> None:
        _make_conn()._validate_params([1, 2])
        _make_conn()._validate_params((1, 2))

    def test_none_accepted(self) -> None:
        _make_conn()._validate_params(None)


def test_validate_timeout_default_min_value_zero_exclusive() -> None:
    validate_timeout(0.001, name="timeout")  # default floor is 0


def test_validate_timeout_below_explicit_min_value_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0.01"):
        validate_timeout(0.001, name="close_timeout", min_value=0.01)


def test_validate_timeout_at_explicit_min_value_accepted() -> None:
    validate_timeout(0.01, name="close_timeout", min_value=0.01)


def test_validate_timeout_above_explicit_min_value_accepted() -> None:
    validate_timeout(0.5, name="close_timeout", min_value=0.01)


def test_validate_timeout_zero_rejected_before_min_value_check() -> None:
    """Zero hits the positive check before the floor; diagnostic mentions
    positivity, not the floor."""
    with pytest.raises(ValueError, match="positive finite number"):
        validate_timeout(0.0, name="close_timeout", min_value=0.01)


def test_validate_timeout_floor_diagnostic_omits_fin_text_without_rationale() -> None:
    """Without ``min_value_rationale=`` the diagnostic stays neutral, so
    non-close-timeout callers do not inherit the FIN-flush explanation."""
    with pytest.raises(ValueError, match=r"close_timeout must be >= 0\.01") as exc:
        validate_timeout(0.001, name="close_timeout", min_value=0.01)
    assert "FIN flushes" not in str(exc.value)
    assert "TIME_WAIT" not in str(exc.value)


def test_validate_timeout_message_mentions_fin_flush_rationale_when_passed() -> None:
    """``min_value_rationale=`` is appended to the diagnostic."""
    with pytest.raises(ValueError, match="FIN flushes"):
        validate_timeout(
            0.001,
            name="close_timeout",
            min_value=0.01,
            min_value_rationale=(
                "Below this floor, the dispose-time writer-close may "
                "complete before FIN flushes, leaving connections "
                "lingering in TIME_WAIT."
            ),
        )


def test_dqlite_connection_close_timeout_below_floor_rejected() -> None:
    """The constructor enforces the floor and passes the FIN-flush rationale."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01") as exc:
        DqliteConnection("localhost:9001", close_timeout=0.001)
    assert "FIN flushes" in str(exc.value), (
        "DqliteConnection must wrap validate_timeout with the "
        "close-timeout-specific FIN-flush rationale so operators "
        "understand the reason for the floor."
    )


def test_connection_pool_close_timeout_below_floor_rejected() -> None:
    """Same for ConnectionPool — also passes the FIN-flush rationale."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01") as exc:
        ConnectionPool(addresses=["localhost:9001"], close_timeout=0.001)
    assert "FIN flushes" in str(exc.value)


def test_dqlite_connection_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works."""
    DqliteConnection("localhost:9001", close_timeout=0.5)


def test_connection_pool_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works."""
    ConnectionPool(addresses=["localhost:9001"], close_timeout=0.5)


@pytest.mark.parametrize("bad", [True, False])
def test_connection_rejects_bool_timeout(bad: bool) -> None:
    with pytest.raises(ValueError, match="bool"):
        DqliteConnection("localhost:9001", timeout=bad)


@pytest.mark.parametrize("bad", [True, False])
def test_connection_rejects_bool_close_timeout(bad: bool) -> None:
    with pytest.raises(ValueError, match="bool"):
        DqliteConnection("localhost:9001", close_timeout=bad)


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_pool_max_attempts_validation_message(bad_value: int) -> None:
    with pytest.raises(ValueError, match="must be at least 1") as exc_info:
        ConnectionPool(["a:9001"], max_attempts=bad_value)
    assert f"got {bad_value}" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [0, -1, -100])
async def test_cluster_max_attempts_validation_message(bad_value: int) -> None:
    cluster = ClusterClient(MemoryNodeStore(["a:9001"]), timeout=1.0)
    with pytest.raises(ValueError, match="must be at least 1") as exc_info:
        await cluster.connect(database="x", max_attempts=bad_value)
    assert f"got {bad_value}" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pool_and_cluster_max_attempts_messages_share_wording() -> None:
    """Both validators emit ``must be at least 1`` plus ``got X``, matchable by one regex."""
    pool_msg: str | None = None
    cluster_msg: str | None = None

    try:
        ConnectionPool(["a:9001"], max_attempts=0)
    except ValueError as e:
        pool_msg = str(e)

    cluster = ClusterClient(MemoryNodeStore(["a:9001"]), timeout=1.0)
    try:
        await cluster.connect(database="x", max_attempts=0)
    except ValueError as e:
        cluster_msg = str(e)

    assert pool_msg is not None and cluster_msg is not None
    for msg in (pool_msg, cluster_msg):
        assert "must be at least 1" in msg
        assert "got 0" in msg
